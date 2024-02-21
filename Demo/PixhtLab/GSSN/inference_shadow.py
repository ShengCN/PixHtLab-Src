import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import argparse
import time
from tqdm import tqdm
import numpy as np
import os
from os.path import join

import math

import cv2
import random

import sys
# sys.path.insert(0, '../../Training/app/models')
sys.path.insert(0, '/home/ysheng/Documents/Research/GSSN/Training/app/models')

from SSN_v1 import SSN_v1
from SSN import SSN


class SSN_Infernece():
    def __init__(self, ckpt, device=torch.device('cuda:0')):
        self.device = device
        self.model  = SSN(3, 1,  mid_act='gelu', out_act='null', resnet=False)

        weight = torch.load(ckpt)
        self.model.to(device)
        self.model.load_state_dict(weight['model'])

        # inference related
        BINs    = 100
        MAX_RAD = 20
        self.size_interval     = MAX_RAD / BINs
        self.soft_distribution = [[np.exp(-0.2 * (i - j) ** 2) for i in np.arange(BINs)] for j in np.arange(BINs)]


    def render_ss(self, input_np, softness):
        """ input_np:
              H x W x C
        """
        input_tensor = torch.tensor(input_np.transpose((2, 0, 1)))[None, ...].float().to(self.device)
        transform    = T.Resize((256, 256))

        c = input_tensor.shape[1]
        # for i in range(c):
        #     print(input_tensor[:, i].min(), input_tensor[:, i].max())

        # print('softness: ', softness)
        l = torch.from_numpy(np.array(self.soft_distribution[int(softness/self.size_interval)]).astype(np.float32)).unsqueeze(dim=0).to(self.device)

        input_tensor  = transform(input_tensor)
        output_tensor = self.model(input_tensor, l)
        output_np     = output_tensor[0].detach().cpu().numpy().transpose((1,2,0))

        return output_np


if __name__ == '__main__':
    model = SSN_Infernece('weights/0000000700.pt')
