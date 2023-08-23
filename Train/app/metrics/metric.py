import torch
import torch.nn as nn
from . import pytorch_ssim
# import pytorch_ssim

""" Note, current implementation assumes C x H x W
"""

def norm_metric(I1, I2, norm=2):
    return torch.norm(I1-I2, norm)


def RMSE_metric(I1, I2):
    criterion = nn.MSELoss()
    mse = criterion(I1, I2)
    trmse = torch.sqrt(mse)
    return trmse


def RMSE_S_metric(I1, I2):
    """
        Scale Invariant RMSE
        compute loss and alpha for

            min |a*I1 - I2|_2
        return alpha, scale invariant rmse
    """
    d1d1 = I1 * I1
    d1d2 = I1 * I2

    sum_d1d1, sum_d1d2 = d1d1.sum(), d1d2.sum()

    s = 1.0
    if sum_d1d1 > 0.0:
        s = sum_d1d2/sum_d1d1

    return RMSE_metric(s * I1, I2)


ssim_loss_ = pytorch_ssim.SSIM()
def ssim_metric(I1, I2):
    return ssim_loss_(I1, I2)


def PSNR_metric(I1, I2):
    b,c,h,w = I1.shape
    criterion = nn.MSELoss()
    mse = criterion(I1, I2) 
    return 10.0 * torch.log10(1.0/mse)


def ZNCC(I1, I2):
    """
        Zero-normalized cross-correlation (ZNCC)
        https://en.wikipedia.org/wiki/Cross-correlation
    """
    diff = I1 - I2
    B, C, H, W = I1.shape
    num_pixels = H * W

    mu1, mu2 = I1.mean(), I2.mean()
    cen1, cen2 = I1 - mu1, I2 - mu2

    sig1 = (cen1 * cen1 / num_pixels).sum() ** 0.5
    sig2 = (cen2 * cen2 / num_pixels).sum() ** 0.5

    if sig1 == 0 or sig2 == 0:
        return 0.0

    return (cen1 * cen2).sum() / (sig1 * sig2 * num_pixels)


def compute_all_metric(I1, I2):
    rmse = RMSE_metric(I1, I2)
    rmses = RMSE_S_metric(I1, I2)
    ssim = ssim_metric(I1, I2)
    psnr = PSNR_metric(I1, I2)
    zncc = ZNCC(I1, I2)
    return rmse, rmses, ssim, psnr, zncc

if __name__ == '__main__':
    h, w, c = 256, 256, 3
    test1 = torch.zeros(1, c, h, w)
    test2 = torch.ones(1, c, h, w)

    test3 = torch.randn(1, c, h, w)
    test4 = torch.randn(1, c, h, w)

    print('n vs a: RMSE \t RMSES \t SSIM \t PSNR \t ZNCC')
    rmse, rmses, ssim, psnr, zncc = compute_all_metric(test1, test1)
    print('0 vs 0: {} \t {} \t {} \t {} \t {}'.format(rmse, rmses, ssim, psnr, zncc))

    rmse, rmses, ssim, psnr, zncc = compute_all_metric(test2, test2)
    print('1 vs 1: {} \t {} \t {} \t {} \t {}'.format(rmse, rmses, ssim, psnr, zncc))

    rmse, rmses, ssim, psnr, zncc = compute_all_metric(test1, test2)
    print('0 vs 1: {} \t {} \t {} \t {} \t {}'.format(rmse, rmses, ssim, psnr, zncc))

    rmse, rmses, ssim, psnr, zncc = compute_all_metric(test3, test3)
    print('random vs self: {} \t {} \t {} \t {} \t {}'.format(rmse, rmses, ssim, psnr, zncc))

    rmse, rmses, ssim, psnr, zncc = compute_all_metric(test3, test4)
    print('random vs random: {} \t {} \t {} \t {} \t {}'.format(rmse, rmses, ssim, psnr, zncc))
