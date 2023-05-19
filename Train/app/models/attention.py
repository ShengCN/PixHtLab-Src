from inspect import isfunction
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from .blocks import get_norm, zero_module


def QKV_Attention(qkv, num_heads):
    """
    Apply QKV attention.
    :param qkv: an [N x (3 * C) x T] tensor of Qs, Ks, and Vs.
    :return: an [N x H' x T] tensor after attention.
    """
    B, C, HW = qkv.shape
    if C % 3 != 0:
        raise ValueError('QKV shape is wrong: {}, {}, {}'.format(B, C, HW))

    split_size = C // (3 * num_heads)
    q, k, v = qkv.chunk(3, dim=1)
    scale      = 1.0/math.sqrt(math.sqrt(split_size))
    weight = torch.einsum('bct, bcs->bts',
                          (q * scale).view(B * num_heads, split_size, HW),
                          (k * scale).view(B * num_heads, split_size, HW))

    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    ret    = torch.einsum("bts,bcs->bct", weight, v.reshape(B * num_heads, split_size, HW))

    return ret.reshape(B, -1, HW)


class AttentionBlock(nn.Module):
    """
        https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
        https://github.com/whai362/PVT/blob/a24ba02c249a510581a84f821c26322534b03a10/detection/pvt_v2.py#L57
    """

    def __init__(self, in_channels, num_heads, qkv_bias=False, sr_ratio=1, linear=True):
        super().__init__()

        self.num_heads = num_heads
        self.norm = get_norm(in_channels, 'Group')
        self.qkv  = nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 3, kernel_size = 1)

        self.proj = zero_module(nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size = 1))


    def forward(self, x):
        b, c, *spatial = x.shape
        num_heads = self.num_heads

        x   = x.reshape(b, c, -1) # B x C x HW
        x   = self.norm(x)
        qkv = self.qkv(x) # b x c x HW ->  B x 3C x HW
        h   = QKV_Attention(qkv, num_heads)
        h   = self.proj(h)

        return (x + h).reshape(b,c,*spatial) # additive attention, similar to ResNet?



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


if __name__ == '__main__':
    model = AttentionBlock(in_channels=256, num_heads=8)

    x = torch.randn(5, 256, 32, 32, dtype=torch.float32)
    y = model(x)
    print('{}, {}'.format(x.shape, y.shape))

    get_model_size(model)
