import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

def get_activation(activation_func):
    act_func = {
            "relu":nn.ReLU(),
            "sigmoid":nn.Sigmoid(),
            "prelu":nn.PReLU(num_parameters=1),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=False),
            "gelu":nn.GELU()
            }

    if activation_func is None:
        return nn.Identity()

    if activation_func not in act_func.keys():
        raise ValueError("activation function({}) is not found".format(activation_func))

    activation = act_func[activation_func]
    return activation


def get_layer_info(out_channels, activation_func='relu'):
    #act_func = {"relu":nn.ReLU(), "sigmoid":nn.Sigmoid(), "prelu":nn.PReLU(num_parameters=out_channels)}

    # norm_layer = nn.BatchNorm2d(out_channels, momentum=0.9)
    if out_channels >= 32:
        groups = 32
    else:
        groups = 1

    norm_layer = nn.GroupNorm(groups, out_channels)
    activation = get_activation(activation_func)
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
                 style=False,
                 resnet=True):
        super().__init__()

        self.style = style
        norm_layer, act_func = get_layer_info(in_channels, activation)

        if resnet and in_channels == out_channels:
            self.resnet = True
        else:
            self.resnet = False

        if style:
            self.styleconv = Conv2DMod(in_channels, out_channels, kernel_size)
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.norm = norm_layer
            self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias)
            self.act  = act_func

    def forward(self, x, style_fea=None):
        if self.style:
            res = self.styleconv(x, style_fea)
            res = self.relu(res)
        else:
            h = self.conv(self.act(self.norm(x)))
            if self.resnet:
                res = h + x
            else:
                res = h

        return res


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class Up(nn.Module):
    """ Upscaling then conv """

    def __init__(self, in_channels, out_channels, activation='relu', resnet=True):
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
    def __init__(self,
                 out_channels=3,
                 mid_act='relu',
                 out_act='sigmoid',
                 resnet = True):

        super(Decoder, self).__init__()

        input_channel = 512
        fea_dim       = 100

        self.to_style1 = nn.Linear(in_features=fea_dim, out_features=input_channel)

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
        self.out_conv   = Conv(64, out_channels, activation=mid_act)

        self.out_act = get_activation(out_act)


    def forward(self, x):
        x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1 = x

        y = self.up_16_16_1(x11)

        y = torch.cat((x10, y), dim=1)
        y = self.up_16_16_2(y)

        y = torch.cat((x9, y), dim=1)
        y = self.up_16_16_3(y)

        y = torch.cat((x8, y), dim=1)
        y = self.up_16_32(y)

        y = torch.cat((x7, y), dim=1)
        y = self.up_32_32_1(y)

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
        y = self.out_act(y)

        return y


class SSN_v1(nn.Module):
    """ Implementation of Relighting Net """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 mid_act='leaky',
                 out_act='sigmoid',
                 resnet=True):
        super(SSN_v1, self).__init__()
        self.encoder = Encoder(in_channels, mid_act=mid_act, resnet=resnet)
        self.decoder = Decoder(out_channels, mid_act=mid_act, out_act=out_act, resnet=resnet)


    def forward(self, x, softness):
        """
            Input is (source image, target light, source light, )
            Output is: predicted new image, predicted source light, self-supervision image
        """
        latent  = self.encoder(x)
        pred    = self.decoder(latent)

        return pred


if __name__ == '__main__':
    test_input = torch.randn(5, 1, 256, 256)
    style = torch.randn(5, 100)

    model = SSN_v1(1, 1, mid_act='gelu', out_act='gelu', resnet=True)
    test_out = model(test_input, style)

    print('Ouptut shape: ', test_out.shape)
