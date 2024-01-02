import os
from os.path import join
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import logging
from time import time
import cv2
import h5py
from .GSSN_Dataset import *


class GSSN_Testing_Dataset(Dataset):
    def __init__(self, opt: dict, is_training:bool):
        self.is_training    = is_training
        self.transform      = transforms.Compose([transforms.ToTensor()])
        self.input_type     = opt['type']
        self.df_hdf5        = h5py.File(opt['hdf5_file'], 'r')

        self.total_data_len = self.df_hdf5['{}/{}'.format('train', CUR_H5_DS)].shape[0]

        self.ignore_shading = opt['ignore_shading']

        logging.info('Use hdf5. Test: {}'.format(self.total_data_len))


    def __len__(self):
        return self.total_data_len


    def __getitem__(self, idx):
        try:
            if self.ignore_shading:
                x, softness, y, light = hdf52data(self.df_hdf5, idx, self.input_type, ignore_shading=True, is_training=True)

                return {
                    'x':{'x':self.transform(x).float(), 'softness':softness.astype(np.float32)},
                    'y':self.transform(y).float(),
                    'light':light }
            else:
                x, softness, y, light, shading = hdf52data(self.df_hdf5, idx, self.input_type, ignore_shading=False)
                shading = torch.from_numpy(shading)

                return {
                    'x':(self.transform(x).float(), softness.astype(np.float32)),
                    'y':self.transform(y).float(),
                    'light':light,
                    'shading':shading.float()}

        except BaseException as err:
            logging.error('File {} has problem: {}'.format(idx, err))
            self.df_hdf5.close()


if __name__ == '__main__':
    opt = {'dataset': {'type': 'Buffer_Channel',
                       'hdf5_file': 'Dataset1/split_face/test/WALL_DS/dataset.hdf5',
                       'rech_grad': False,
                       'ignore_shading': True}}
    ds = GSSN_Testing_Dataset(opt['dataset'], True)

    for i, d in enumerate(tqdm(ds, desc='testing dataset')):
        x, y = d['x'], d['y']

    ds_loader = DataLoader(ds, num_workers=32, batch_size=32)
    for i, d in enumerate(tqdm(ds_loader, total=len(ds_loader))):
        x, y = d['x'], d['y']
