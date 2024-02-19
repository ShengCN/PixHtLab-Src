import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
from tqdm import tqdm
import numpy as np
import os
from os.path import join

import math

import cv2
import random
from .ssn.random_pattern import random_pattern
from .ssn.ssn import Relight_SSN
device = torch.device('cuda:0')

def net_render_np(model, mask_np, hard_shadow_np, size, orientation):
    """
    input:
        mask_np shape: b x c x h x w
        ibl_np shape: 1 x 16 x 32
    output:
        shadow_predict shape: b x c x h x w
    """

    size_interval = 0.5 / 100
    ori_interval = np.pi / 100

    soft_distribution = [[np.exp(-0.2 * (i - j) ** 2) for i in np.arange(0.5 / size_interval)]
                         for j in np.arange(0.5 / size_interval)]

    # print('mask_np: {}, hard_shadow_np: {}'.format(mask_np.shape, hard_shadow_np.shape))
    s = time.time()
    if mask_np.dtype == np.uint8:
        mask_np = mask_np / 255.0

    mask, h_shadow = torch.Tensor(mask_np), torch.Tensor(hard_shadow_np)
    size_soft = torch.Tensor(np.array(soft_distribution[int(size / size_interval)])).unsqueeze(0)
    ori_soft = torch.Tensor(np.array(soft_distribution[int(orientation / ori_interval)])).unsqueeze(0)

    with torch.no_grad():
        I_m, I_h, size_t, ori = mask.to(device), h_shadow.to(device), size_soft.to(device), ori_soft.to(device)
        # print('I_m: {}, I_h: {}'.format(I_m.shape, I_h.shape))
        predicted_img = model(I_h, I_m, size_t, ori)

    # print('net predict finished, time: {}s'.format(time.time() - s))

    return predicted_img.detach().cpu().numpy()

def init_models(ckpt):
    baseline_model = Relight_SSN(1, 1, is_training=False)
    baseline_checkpoint = torch.load(ckpt)
    baseline_model.to(device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    return baseline_model

if __name__ == '__main__':
    softness = [0.02, 0.2, 0.3, 0.4]
    model = init_models('weights/human_baseline123.pt')
    mask, hard_shadow, size, orientation = np.random.randn(1,1,256,256), np.random.randn(1,1,256,256), softness[0], 0 
    shadow = net_render_np(model, mask, hard_shadow, size, orientation)
