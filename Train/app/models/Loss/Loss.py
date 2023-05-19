import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

import numpy as np
import cv2

# from vgg19_loss import VGG19Loss
# import pytorch_ssim

from .vgg19_loss import VGG19Loss
from . import pytorch_ssim
from abc import ABC, abstractmethod
from collections import OrderedDict

class abs_loss(ABC):
    def loss(self, gt_img, pred_img):
        pass


class norm_loss(abs_loss):
    def __init__(self, norm=1):
        self.norm = norm


    def loss(self, gt_img, pred_img):
        """ M * (I-I') """
        b, c, h, w = gt_img.shape
        return torch.norm(gt_img-pred_img, self.norm)/(h * w * b)



class ssim_loss(abs_loss):
    def __init__(self, window_size=11, channel=1):
        """ Let's try mean ssim!
        """
        self.channel     = channel
        self.window_size = window_size
        self.window      = self.create_mean_window(window_size, channel)


    def loss(self, gt_img, pred_img):
        b, c, h, w = gt_img.shape
        if c != self.channel:
            self.channel = c
            self.window = self.create_mean_window(self.window_size, self.channel)

        self.window = self.window.to(gt_img).type_as(gt_img)
        l = 1.0 - self.ssim_compute(gt_img, pred_img)
        return l


    def create_mean_window(self, window_size, channel):
        window = Variable(torch.ones(channel, 1, window_size, window_size).float())
        window = window/(window_size * window_size)
        return window


    def ssim_compute(self, gt_img, pred_img):
        window      = self.window
        window_size = self.window_size
        channel     = self.channel

        mu1 = F.conv2d(gt_img, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(pred_img, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(gt_img*gt_img, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(pred_img*pred_img, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12   = F.conv2d(gt_img*pred_img, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()


class hierarchical_ssim_loss(abs_loss):
    def __init__(self, patch_list: list):
        self.ssim_loss_list = [pytorch_ssim.SSIM(window_size=ws) for ws in patch_list]


    def loss(self, gt_img, pred_img):
        b, c, h, w = gt_img.shape
        total_loss = 0.0
        for loss_func in self.ssim_loss_list:
            total_loss +=  (1.0-loss_func(gt_img, pred_img))

        return total_loss/b


class vgg_loss(abs_loss):
    def __init__(self):
        self.vgg19_ = VGG19Loss()


    def loss(self, gt_img, pred_img):
        b, c, h, w = gt_img.shape
        v = self.vgg19_(gt_img, pred_img, pred_img.device)
        return v/b


class grad_loss(abs_loss):
    def __init__(self, k=4):
        self.k = 4

    def loss(self, disp_img, rgb_img=None):
        """ Note, gradient loss should be weighted by an edge-aware weight
        """
        b, c, h, w = disp_img.shape

        grad_loss = 0.0
        for i in range(self.k):
            div_factor               = 2 ** i
            cur_transform            = T.Resize([h // div_factor, ])
            # cur_diff                 = cur_transform(diff)
            # cur_diff_dx, cur_diff_dy = self.img_grad(cur_diff)
            cur_disp = cur_transform(disp_img)

            cur_disp_dx, cur_disp_dy = self.img_grad(cur_disp)

            if rgb_img is not None:
                cur_rgb  = cur_transform(rgb_img)
                cur_rgb_dx, cur_rgb_dy = self.img_grad(cur_rgb)

                cur_rgb_dx = torch.exp(-torch.mean(torch.abs(cur_rgb_dx), dim=1, keepdims=True))
                cur_rgb_dy = torch.exp(-torch.mean(torch.abs(cur_rgb_dy), dim=1, keepdims=True))
                grad_loss += (torch.sum(torch.abs(cur_disp_dx) * cur_rgb_dx) + torch.sum(torch.abs(cur_disp_dy) * cur_rgb_dy)) / (h * w * self.k)
            else:
                grad_loss += (torch.sum(torch.abs(cur_disp_dx)) + torch.sum(torch.abs(cur_disp_dy))) / (h * w * self.k)

        return grad_loss/b


    def gloss(self, gt, pred):
        """ Loss on the gradient domain
        """
        b, c, h, w = gt.shape
        gt_dx, gt_dy = self.img_grad(gt)
        pred_dx, pred_dy = self.img_grad(pred)

        loss = (gt_dx-pred_dx) ** 2 + (gt_dy - pred_dy) ** 2
        return loss.sum()/(b * h * w)


    def laploss(self, pred):
        b, c, h, w = pred.shape
        lap = self.img_laplacian(pred)

        return torch.abs(lap).sum()/(b * h * w)


    def img_laplacian(self, img):
        b, c, h, w = img.shape
        laplacian  = torch.tensor([[1, 4, 1], [4, -20, 4], [1, 4, 1]])

        laplacian_kernel = laplacian.float().unsqueeze(0).expand(1, c, 3, 3).to(img)

        lap = F.conv2d(img, laplacian_kernel, padding=1, stride=1)
        return lap


    def img_grad(self, img):
        """ Comptue image gradient by sobel filtering
            img: B x C x H x W
        """

        b, c, h, w = img.shape
        ysobel     = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        xsobel     = ysobel.transpose(0,1)

        xsobel_kernel = xsobel.float().unsqueeze(0).expand(1, c, 3, 3).to(img)
        ysobel_kernel = ysobel.float().unsqueeze(0).expand(1, c, 3, 3).to(img)
        dx = F.conv2d(img, xsobel_kernel, padding=1, stride=1)
        dy = F.conv2d(img, ysobel_kernel, padding=1, stride=1)

        return dx, dy



class sharp_loss(abs_loss):
    """  Sharpness term
            1. laplacian
            2. image contrast
            3. image variance
    """
    def __init__(self, window_size=11, channel=1):
        self.window_size = window_size
        self.channel     = channel
        self.window      = self.create_mean_window(window_size, self.channel)


    def loss(self, gt_img, pred_img):
        """ Note, gradient loss should be weighted by an edge-aware weight
        """
        b, c, h, w = gt_img.shape

        if c != self.channel:
            self.channel = c
            self.window = self.create_mean_window(self.window_size, self.channel)

        self.window = self.window.to(gt_img).type_as(gt_img)

        channel     = self.channel
        window      = self.window
        window_size = self.window_size

        mu1 = F.conv2d(gt_img, window, padding = window_size//2, groups = channel)  + 1e-6
        mu2 = F.conv2d(pred_img, window, padding = window_size//2, groups = channel) + 1e-6

        constrast1 = torch.absolute((gt_img - mu1)/mu1)
        constrast2 = torch.absolute((pred_img - mu2)/mu2)

        variance1 = (gt_img-mu1) ** 2
        variance2 = (pred_img-mu2) ** 2

        laplacian1 = self.img_laplacian(gt_img)
        laplacian2 = self.img_laplacian(pred_img)

        S1 = -laplacian1 - constrast1 - variance1
        S2 = -laplacian2 - constrast2 - variance2

        # import pdb; pdb.set_trace()
        total = torch.absolute(S1-S2).mean()
        return total


    def img_laplacian(self, img):
        b, c, h, w = img.shape
        laplacian  = torch.tensor([[1, 4, 1], [4, -20, 4], [1, 4, 1]])

        laplacian_kernel = laplacian.float().unsqueeze(0).expand(1, c, 3, 3).to(img)

        lap = F.conv2d(img, laplacian_kernel, padding=1, stride=1)
        return lap


    def create_mean_window(self, window_size, channel):
        window = Variable(torch.ones(channel, 1, window_size, window_size).float())
        window = window/(window_size * window_size)
        return window


if __name__ == '__main__':
    a = torch.rand(3,3,128,128)
    b = torch.rand(3,3,128,128)

    ssim = ssim_loss()
    loss = ssim.loss(a, b)
    print(loss.shape, loss)

    loss = ssim.loss(a, a)
    print(loss.shape, loss)

    loss = ssim.loss(b, b)
    print(loss.shape, loss)

    grad = grad_loss()
    loss = grad.loss(a, [b, b])
    print(loss.shape, loss)

    sharp = sharp_loss()
    loss = sharp.loss(a, b)
    print(loss.shape, loss)
