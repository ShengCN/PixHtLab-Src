import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from .SSN import  Conv, Conv2DMod, Decoder, Up
from .attention import AttentionBlock
from .blocks import ResBlock, Res_Type, get_activation

class Attention_Encoder(nn.Module):
    def __init__(self, in_channels=3, mid_act='gelu', dropout=0.0, num_heads=8, resnet=True):
        super(Attention_Encoder, self).__init__()

        self.in_conv        = Conv(in_channels, 32-in_channels, stride=1, activation=mid_act, resnet=resnet)
        self.down_32_64     = Conv(32, 64, stride=2, activation=mid_act, resnet=resnet)
        self.down_64_64_1   = Conv(64, 64, activation=mid_act, resnet=resnet)

        self.down_64_128    = Conv(64, 128, stride=2, activation=mid_act, resnet=resnet)
        self.down_128_128_1 = Conv(128, 128,  activation=mid_act, resnet=resnet)

        self.down_128_256   = Conv(128, 256, stride=2, activation=mid_act, resnet=resnet)
        self.down_256_256_1 = Conv(256, 256, activation=mid_act, resnet=resnet)
        self.down_256_256_1_attn = AttentionBlock(256, num_heads)

        self.down_256_512   = Conv(256, 512, stride=2, activation=mid_act, resnet=resnet)
        self.down_512_512_1 = Conv(512, 512, activation=mid_act, resnet=resnet)
        self.down_512_512_1_attn = AttentionBlock(512, num_heads)

        self.down_512_512_2 = Conv(512, 512, activation=mid_act, resnet=resnet)
        self.down_512_512_2_attn = AttentionBlock(512, num_heads)

        self.down_512_512_3 = Conv(512, 512, activation=mid_act, resnet=resnet)
        self.down_512_512_3_attn = AttentionBlock(512, num_heads)


    def forward(self, x):
        x1 = self.in_conv(x)  # 32 x 256 x 256
        x1 = torch.cat((x, x1), dim=1)

        x2 = self.down_32_64(x1)
        x3 = self.down_64_64_1(x2)

        x4 = self.down_64_128(x3)
        x5 = self.down_128_128_1(x4)

        x6 = self.down_128_256(x5)
        x7 = self.down_256_256_1(x6)
        x7 = self.down_256_256_1_attn(x7)

        x8 = self.down_256_512(x7)
        x9 = self.down_512_512_1(x8)
        x9 = self.down_512_512_1_attn(x9)

        x10 = self.down_512_512_2(x9)
        x10 = self.down_512_512_2_attn(x10)

        x11 = self.down_512_512_3(x10)
        x11 = self.down_512_512_3_attn(x11)

        return x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1


class Attention_Decoder(nn.Module):
    def __init__(self, out_channels=3, mid_act='gelu', out_act='sigmoid', resnet = True, num_heads=8):

        super(Attention_Decoder, self).__init__()

        input_channel = 512
        fea_dim       = 100

        self.to_style1 = nn.Linear(in_features=fea_dim, out_features=input_channel)

        self.up_16_16_1 = Conv(input_channel, 256, activation=mid_act, style=False, resnet=resnet)
        self.up_16_16_1_attn = AttentionBlock(256, num_heads=num_heads)

        self.up_16_16_2 = Conv(768, 512, activation=mid_act, resnet=resnet)
        self.up_16_16_2_attn = AttentionBlock(512, num_heads=num_heads)

        self.up_16_16_3      = Conv(1024, 512, activation=mid_act, resnet=resnet)
        self.up_16_16_3_attn = AttentionBlock(512, num_heads=num_heads)

        self.up_16_32        = Up(1024, 256, activation=mid_act, resnet=resnet)
        self.to_style2       = nn.Linear(in_features=fea_dim, out_features=512)
        self.up_32_32_1      = Conv(512, 256, activation=mid_act, style=False, resnet=resnet)
        self.up_32_32_1_attn = AttentionBlock(256, num_heads=num_heads)

        self.up_32_64   = Up(512, 128, activation=mid_act, resnet=resnet)
        self.to_style3  = nn.Linear(in_features=fea_dim, out_features=256)
        self.up_64_64_1 = Conv(256, 128, activation=mid_act, style=False, resnet=resnet)

        self.up_64_128    = Up(256, 64, activation=mid_act, resnet=resnet)
        self.to_style4    = nn.Linear(in_features=fea_dim, out_features=128)
        self.up_128_128_1 = Conv(128, 64, activation=mid_act, style=False, resnet=resnet)

        self.up_128_256 = Up(128, 32, activation=mid_act, resnet=resnet)
        self.out_conv   = Conv(64, out_channels, activation=out_act)
        self.out_act = get_activation(out_act)


    def forward(self, x):
        x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1 = x

        y = self.up_16_16_1(x11)  # 256 x 16 x 16
        y = self.up_16_16_1_attn(y)

        y = torch.cat((x10, y), dim=1)  # 768 x 16 x 16
        y = self.up_16_16_2(y, y)  # 512 x 16 x 16
        y = self.up_16_16_2_attn(y)


        y = torch.cat((x9, y), dim=1)  # 1024 x 16 x 16
        y = self.up_16_16_3(y, y)  # 512 x 16 x 16
        y = self.up_16_16_3_attn(y)

        y = torch.cat((x8, y), dim=1)  # 1024 x 16 x 16
        y = self.up_16_32(y, y)  # 256 x 32 x 32

        y = torch.cat((x7, y), dim=1)
        y = self.up_32_32_1(y)  # 256 x 32 x 32
        y = self.up_32_32_1_attn(y)

        y = torch.cat((x6, y), dim=1)
        y = self.up_32_64(y, y)

        y = torch.cat((x5, y), dim=1)

        y = self.up_64_64_1(y)  # 128 x 64 x 64

        y = torch.cat((x4, y), dim=1)
        y = self.up_64_128(y, y)

        y = torch.cat((x3, y), dim=1)
        y = self.up_128_128_1(y)  # 64 x 128 x 128

        y = torch.cat((x2, y), dim=1)
        y = self.up_128_256(y, y)  # 32 x 256 x 256

        y = torch.cat((x1, y), dim=1)
        y = self.out_conv(y, y)  # 3 x 256 x 256
        y = self.out_act(y)
        return y


class Attention_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, resnet=True, mid_act='gelu', out_act='gelu'):
        super(Attention_Unet, self).__init__()
        self.encoder = Attention_Encoder(in_channels, mid_act, num_heads, resnet)
        self.decoder = Attention_Decoder(out_channels, mid_act, out_act, resnet)


    def forward(self, x):
        latent  = self.encoder(x)
        pred    = self.decoder(latent)
        return pred


if __name__ == '__main__':
    test_input = torch.randn(5, 1, 256, 256)
    style = torch.randn(5, 100)

    model = SSN_v1(1, 1, mid_act='gelu', out_act='gelu', resnet=True)
    test_out = model(test_input, style)

    print('Ouptut shape: ', test_out.shape)
