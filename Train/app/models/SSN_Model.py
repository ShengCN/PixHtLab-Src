import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

def weights_init(init_type='gaussian', std=0.02):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, std)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True

def get_optimizer(opt, model):
    lr           = float(opt['hyper_params']['lr'])
    beta1        = float(opt['model']['beta1'])
    weight_decay = float(opt['model']['weight_decay'])
    opt_name     = opt['model']['optimizer']

    optim_params = []
    # weight decay
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue  # frozen weights

        if key[-4:] == 'bias':
            optim_params += [{'params': value, 'weight_decay': 0.0}]
        else:
            optim_params += [{'params': value,
                              'weight_decay': weight_decay}]

    if opt_name == 'Adam':
        return optim.Adam(optim_params,
                            lr=lr,
                            betas=(beta1, 0.999),
                            eps=1e-5)
    else:
        err = '{} not implemented yet'.format(opt_name)
        logging.error(err)
        raise NotImplementedError(err)


def get_activation(activation):
    if activation is None:
        return nn.Identity()

    act_func = {
        'relu':nn.ReLU(),
        'sigmoid':nn.Sigmoid(),
        'tanh':nn.Tanh(),
        'prelu':nn.PReLU(),
        'leaky':nn.LeakyReLU(0.2),
        'gelu':nn.GELU(),
        }
    if activation not in act_func.keys():
        logging.error("activation {} is not implemented yet".format(activation))
        assert False

    return act_func[activation]

def get_norm(out_channels, norm_type='Instance'):
    norm_set = ['Instance', 'Batch', 'Group']
    if norm_type not in norm_set:
        err = "Normalization {} has not been implemented yet"
        logging.error(err)
        raise ValueError(err)

    if norm_type == 'Instance':
        return nn.InstanceNorm2d(out_channels, affine=True)

    if norm_type == 'Batch':
        return nn.BatchNorm2d(out_channels)

    if norm_type == 'Group':
        if out_channels >= 32:
            groups = 32
        else:
            groups = 1

        return nn.GroupNorm(groups, out_channels)

    else:
        raise NotImplementedError('{} has not implemented yet'.format(norm_type))



def get_layer_info(out_channels, activation_func='relu'):
    activation = get_activation(activation_func)
    norm_layer = get_norm(out_channels, 'Group')
    return norm_layer, activation


class Conv(nn.Module):
    """ (convolution => [BN] => ReLU) """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 activation='leaky',
                 resnet=True):
        super().__init__()

        norm_layer, act_func = get_layer_info(out_channels,activation)

        if resnet and in_channels == out_channels:
            self.resnet = True
        else:
            self.resnet = False

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias),
            norm_layer,
            act_func)

    def forward(self, x):
        res = self.conv(x)

        if self.resnet:
            res = res + x

        return res



class Up(nn.Module):
    """ Upscaling then conv """

    def __init__(self, in_channels, out_channels, activation='relu',  resnet=True):
        super().__init__()

        self.up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up       = Conv(in_channels, out_channels, activation=activation, resnet=resnet)

    def forward(self, x):
        x = self.up_layer(x)
        return self.up(x)



class DConv(nn.Module):
    """ Double Conv Layer
    """
    def __init__(self, in_channels, out_channels, activation='relu', resnet=True):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels, activation=activation, resnet=resnet)
        self.conv2 = Conv(out_channels, out_channels, activation=activation, resnet=resnet)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Encoder(nn.Module):
    def __init__(self, in_channels=3, mid_act='leaky', resnet=True):
        super(Encoder, self).__init__()
        self.in_conv        = Conv(in_channels, 32-in_channels, stride=1, activation=mid_act, resnet=resnet)
        self.down_32_64     = Conv(32, 64, stride=2, activation=mid_act, resnet=resnet)
        self.down_64_64_1   = Conv(64, 64, activation=mid_act, resnet=resnet)
        self.down_64_128    = Conv(64, 128, stride=2, activation=mid_act, resnet=resnet)
        self.down_128_128_1 = Conv(128, 128,  activation=mid_act, resnet=resnet)
        self.down_128_256   = Conv(128, 256, stride=2, activation=mid_act, resnet=resnet)
        self.down_256_256_1 = Conv(256, 256, activation=mid_act, resnet=resnet)
        self.down_256_512   = Conv(256, 512, stride=2, activation=mid_act, resnet=resnet)
        self.down_512_512_1 = Conv(512, 512, activation=mid_act, resnet=resnet)
        self.down_512_512_2 = Conv(512, 512, activation=mid_act, resnet=resnet)
        self.down_512_512_3 = Conv(512, 512, activation=mid_act, resnet=resnet)


    def forward(self, x):
        x1 = self.in_conv(x)  # 32 x 256 x 256
        x1 = torch.cat((x, x1), dim=1)

        x2 = self.down_32_64(x1)
        x3 = self.down_64_64_1(x2)

        x4 = self.down_64_128(x3)
        x5 = self.down_128_128_1(x4)

        x6 = self.down_128_256(x5)
        x7 = self.down_256_256_1(x6)

        x8 = self.down_256_512(x7)
        x9 = self.down_512_512_1(x8)
        x10 = self.down_512_512_2(x9)
        x11 = self.down_512_512_3(x10)

        return x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1


class Decoder(nn.Module):
    """ Up Stream Sequence """

    def __init__(self,
                 out_channels=3,
                 mid_act='relu',
                 out_act='sigmoid',
                 resnet = True):

        super(Decoder, self).__init__()

        input_channel = 512
        fea_dim       = 100


        self.up_16_16_1 = Conv(input_channel, 256, activation=mid_act, resnet=resnet)
        self.up_16_16_2 = Conv(768, 512, activation=mid_act, resnet=resnet)
        self.up_16_16_3 = Conv(1024, 512, activation=mid_act, resnet=resnet)

        self.up_16_32   = Up(1024, 256, activation=mid_act, resnet=resnet)
        self.up_32_32_1 = Conv(512, 256, activation=mid_act, resnet=resnet)

        self.up_32_64   = Up(512, 128, activation=mid_act, resnet=resnet)
        self.up_64_64_1 = Conv(256, 128, activation=mid_act, resnet=resnet)

        self.up_64_128    = Up(256, 64, activation=mid_act, resnet=resnet)
        self.up_128_128_1 = Conv(128, 64, activation=mid_act, resnet=resnet)

        self.up_128_256 = Up(128, 32, activation=mid_act, resnet=resnet)
        self.out_conv   = Conv(64, out_channels, activation=out_act)


    def forward(self, x, ibl):
        x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1 = x

        h,w = x10.shape[2:]
        y = ibl.view(-1, 512, 1, 1).repeat(1, 1, h, w)

        y = self.up_16_16_1(y)  # 256 x 16 x 16

        y = torch.cat((x10, y), dim=1)  # 768 x 16 x 16
        y = self.up_16_16_2(y)  # 512 x 16 x 16


        y = torch.cat((x9, y), dim=1)  # 1024 x 16 x 16
        y = self.up_16_16_3(y)  # 512 x 16 x 16

        y = torch.cat((x8, y), dim=1)  # 1024 x 16 x 16
        y = self.up_16_32(y)  # 256 x 32 x 32

        y = torch.cat((x7, y), dim=1)
        y = self.up_32_32_1(y)  # 256 x 32 x 32

        y = torch.cat((x6, y), dim=1)
        y = self.up_32_64(y)

        y = torch.cat((x5, y), dim=1)
        y = self.up_64_64_1(y)  # 128 x 64 x 64

        y = torch.cat((x4, y), dim=1)
        y = self.up_64_128(y)

        y = torch.cat((x3, y), dim=1)
        y = self.up_128_128_1(y)  # 64 x 128 x 128

        y = torch.cat((x2, y), dim=1)
        y = self.up_128_256(y)  # 32 x 256 x 256

        y = torch.cat((x1, y), dim=1)
        y = self.out_conv(y)  # 3 x 256 x 256

        return y


class SSN_Model(nn.Module):
    """ Implementation of Relighting Net """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 mid_act='leaky',
                 out_act='sigmoid',
                 resnet=True):
        super(SSN_Model, self).__init__()

        self.out_act = out_act

        self.encoder = Encoder(in_channels, mid_act=mid_act, resnet=resnet)
        self.decoder = Decoder(out_channels, mid_act=mid_act, out_act=out_act, resnet=resnet)

        # init weights 
        init_func = weights_init('gaussian', std=1e-3)
        self.encoder.apply(init_func)
        self.decoder.apply(init_func)


    def forward(self, x, ibl):
        """
            Input is (source image, target light, source light, )
            Output is: predicted new image, predicted source light, self-supervision image
        """
        latent  = self.encoder(x)
        pred    = self.decoder(latent, ibl)

        if self.out_act == 'sigmoid':
            pred = pred * 30.0

        return pred


if __name__ == '__main__':
    x = torch.randn(5,1,256,256)
    ibl = torch.randn(5, 1, 32, 16)
    model = SSN_Model(1,1)

    y = model(x, ibl)

    print('Output: ', y.shape)
