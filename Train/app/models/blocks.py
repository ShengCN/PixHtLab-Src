from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
    # return param_size + buffer_size
    return size_all_mb


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
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
    act_func = {
        'relu':nn.ReLU(),
        'sigmoid':nn.Sigmoid(),
        'tanh':nn.Tanh(),
        'prelu':nn.PReLU(),
        'leaky_relu':nn.LeakyReLU(0.2),
        'gelu':nn.GELU(),
        }
    if activation not in act_func.keys():
        logging.error("activation {} is not implemented yet".format(activation))
        assert False

    return act_func[activation]


def get_norm(out_channels, norm_type='Group', groups=32):
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
            groups = max(out_channels // 2, 1)

        return nn.GroupNorm(groups, out_channels)
    else:
        raise NotImplementedError


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_type='Batch', activation='relu'):
        super().__init__()

        act_func   = get_activation(activation)
        norm_layer = get_norm(out_channels, norm_type)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='reflect'),
            norm_layer,
            act_func)

    def forward(self, x):
        return self.conv(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Up(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')


class Down(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv

        if self.use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        return self.op(x)


class Res_Type(Enum):
    UP   = 1
    DOWN = 2
    SAME = 3


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout=0.0, updown=Res_Type.DOWN, mid_act='leaky'):
        """ ResBlock to cover several cases:
              1. Up/Down/Same
              2. in_channels != out_channels
        """
        super().__init__()

        self.updown = updown

        self.in_norm = get_norm(out_channels, 'Group')
        self.in_act  = get_activation(mid_act)
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)

        # up down
        if self.updown == Res_Type.DOWN:
            self.h_updown = Down(in_channels, use_conv=True)
            self.x_updown = Down(in_channels, use_conv=True)
        elif self.updown == Res_Type.UP:
            self.h_updown = Up()
            self.x_updown = Up()
        else:
            self.h_updown = nn.Identity()

        self.out_layer = nn.Sequential(
            get_norm(out_channels, 'Group'),
            get_activation(mid_act),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True))
        )


    def forward(self, x):
        # in layer
        h = self.in_act(self.in_norm(x))
        h = self.in_conv(self.h_updown(h))
        x = self.x_updown(x)

        # out layer
        h = self.out_layer(h)
        return x + h



if __name__ == '__main__':
    x = torch.randn(5, 3, 256, 256)
    up = Up()
    conv_down = Down(3, True)
    pool_down = Down(3, False)

    print('Up: {}'.format(up(x).shape))
    print('Conv down: {}'.format(conv_down(x).shape))
    print('Pool down: {}'.format(pool_down(x).shape))

    up_model = ResBlock(3, 6, updown=True)
    down_model = ResBlock(3, 6, updown=False)

    print('model down: {}'.format(up_model(x).shape))
    print('model down: {}'.format(down_model(x).shape))
