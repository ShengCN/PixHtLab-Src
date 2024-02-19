import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssn_submodule import Conv, Up, Up_Stream, get_layer_info, add_coords
import copy

class Relight_SSN(nn.Module):
    """ Implementation of Relighting Net """

    def __init__(self, n_channels=3, out_channels=3, is_training=True, activation_func = 'relu'):
        super(Relight_SSN, self).__init__()
        self.is_training = is_training 

        norm_layer1, activation_func1 = get_layer_info(16, activation_func)
        norm_layer2, activation_func2 = get_layer_info(16, activation_func)
        if norm_layer1 is not None:
            self.in_conv1 = nn.Sequential(
                nn.Conv2d(n_channels, 16, kernel_size=7, padding=3, bias=True),
                norm_layer1,
                activation_func1
            )
        elif norm_layer1 is None:
            self.in_conv1 = nn.Sequential(
                nn.Conv2d(n_channels, 16, kernel_size=7, padding=3, bias=True),
                activation_func1
            )

        if norm_layer2 is not None:
            self.in_conv2 = nn.Sequential(
                nn.Conv2d(n_channels, 16, kernel_size=7, padding=3, bias=True),
                norm_layer2,
                activation_func2
            )
        elif norm_layer2 is None:
            self.in_conv2 = nn.Sequential(
                nn.Conv2d(n_channels, 16, kernel_size=7, padding=3, bias=True),
                activation_func2
            )

        self.down_256_128 = Conv(32, 64, conv_stride=2)
        self.down_128_128 = Conv(64, 64, conv_stride=1)
        self.down_128_64 = Conv(64, 128, conv_stride=2)
        self.down_64_64 = Conv(128, 128, conv_stride=1)
        self.down_64_32 = Conv(128, 256, conv_stride=2)
        self.down_32_32 = Conv(256, 256, conv_stride=1)
        self.down_32_16 = Conv(256, 512, conv_stride=2)
        self.down_16_16_1 = Conv(512, 512, conv_stride=1)
        self.down_16_16_2 = Conv(512, 512, conv_stride=1)
        self.down_16_16_3 = Conv(512, 512, conv_stride=1)
        self.to_bottleneck = Conv(512, 2, conv_stride=1)

        self.up_stream = Up_Stream(out_channels)

    """
        Input is (source image, target light, source light, )
        Output is: predicted new image, predicted source light, self-supervision image
    """

    def forward(self, I_h, I_m, size, angle):
        if self.is_training:
            bs, fake_bs, dim = size.shape
            bs, fake_bs, ch, w, h = I_h.shape
            size = size.reshape(bs * fake_bs, dim)
            angle = angle.reshape(bs * fake_bs, dim)
            I_h = I_h.reshape(bs * fake_bs, ch, w, h)
            I_m = I_m.reshape(bs * fake_bs, ch, w, h)
        else:
            size = size
            angle = angle
            I_h = I_h
            I_m = I_m
        style = torch.cat((size, angle), dim=1)
        x1 = self.in_conv1(I_m)  # 29 x 256 x 256
        x2 = self.in_conv2(I_h)

        x1 = torch.cat((x1, x2), dim=1)  # 32 x 256 x 256

        x2 = self.down_256_128(x1, x1)  # 64 x 128 x 128

        x3 = self.down_128_128(x2, x1)  # 64 x 128 x 128

        x4 = self.down_128_64(x3, x1)  # 128 x 64 x 64

        x5 = self.down_64_64(x4, x1)  # 128 x 64 x 64

        x6 = self.down_64_32(x5, x1)  # 256 x 32 x 32

        x7 = self.down_32_32(x6, x1)  # 256 x 32 x 32

        x8 = self.down_32_16(x7, x1)  # 512 x 16 x 16

        x9 = self.down_16_16_1(x8, x1)  # 512 x 16 x 16

        x10 = self.down_16_16_2(x9, x1)  # 512 x 16 x 16

        x11 = self.down_16_16_3(x10, x1)  # 512 x 16 x 16
        ty = self.up_stream(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, style)

        return ty


def baseline_2_tbaseline(model):
    """ change input layer to be two channels
    """
    input_channel = 2
    tbase_inlayer = nn.Sequential(
        nn.Conv2d(input_channel, 32 - input_channel, kernel_size=7, padding=3, bias=True),
        nn.GroupNorm(1, 32 - input_channel),
        nn.ReLU()
    )
    model.in_conv = tbase_inlayer
    return model


def baseline_2_touchloss(model):
    """ change output layer to be two channels
    """
    touchless_outlayer = nn.Sequential(
        nn.Conv2d(64, 2, stride=1, kernel_size=3, padding=1, bias=True),
        nn.GroupNorm(1, 2),
        nn.ReLU()
    )
    model.up_stream.out_conv = touchless_outlayer
    return model


if __name__ == '__main__':
    mask_test, touch_test = torch.zeros((1, 1, 256, 256)), torch.zeros((1, 1, 256, 256))
    ibl = torch.zeros((1, 1, 16, 32))

    I_s = mask_test
    baseline = Relight_SSN(1, 1)
    baseline_output, _ = baseline(I_s, ibl)

    tbaseline = baseline_2_tbaseline(copy.deepcopy(baseline))
    I_s = torch.cat((mask_test, touch_test), axis=1)
    tbaseline_output, _ = tbaseline(I_s, ibl)

    t_loss_baseline = baseline_2_touchloss(copy.deepcopy(baseline))
    I_s = mask_test
    tloss_output, _ = t_loss_baseline(I_s, ibl)

    print('baseline output: ', baseline_output.shape)
    print('tbaseline output: ', tbaseline_output.shape)
    print('tloss output: ', tloss_output.shape)
