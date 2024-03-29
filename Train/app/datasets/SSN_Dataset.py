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
import pickle

import numbergen as ng
import imagen as ig


class SSN_Dataset(Dataset):
    def __init__(self, opt, is_training):

        keys = ['hdf5_file', 'shadow_per_epoch']
        for k in keys:
            assert k in opt, 'opt must have {}'.format(k)

        random.seed(19920208)
        np.random.seed(19920208)

        self.is_training = is_training
        self.transform = transforms.Compose([ transforms.ToTensor() ])

        hdf5_file        = opt['hdf5_file']
        self.shadow_per_epoch = opt['shadow_per_epoch']

        assert os.path.exists(hdf5_file), 'File {} does not exist'.format(hdf5_file)

        try:
            self.hdf5    = h5py.File(hdf5_file, 'r')
        except BaseException as err:
            logging.error('{}: {}'.format(hdf5_file, err))

        self.random_pattern_generator = random_pattern()
        # block WARNING
        root_logger = logging.getLogger('param')
        root_logger.setLevel(logging.ERROR)

        # prepare meta data
        self.meta, self.mask_ds, self.base_ds = self.prepare_meta(self.hdf5)

        # import pdb; pdb.set_trace()
        random.shuffle(self.meta)
        percentage = 0.9
        self.train_keys = self.meta[:int(len(self.meta)*percentage)]
        self.valid_keys = self.meta[int(len(self.meta)*percentage):]
        # self.train_keys = self.valid_keys = self.meta

        self.keys = self.train_keys if self.is_training else self.valid_keys
        print('Train/Valid: {}/{}'.format(len(self.train_keys), len(self.valid_keys)))

        self.mask_buffer = {}
        self.shadow_buffer = {}

        dataset_root = os.path.dirname(hdf5_file)
        cache_file = join(dataset_root, 'cache/{}.pkl'.format(os.path.basename(hdf5_file))) 
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        if not os.path.exists(cache_file):
            for k in tqdm(self.keys, desc='Loading data'):
                self.mask_buffer[k] = self.mask_ds[k][...]
                self.shadow_buffer[k] = self.base_ds[k][...]
            
            pickle.dump([self.mask_buffer, self.shadow_buffer], open(cache_file, 'wb'))
        else:
            self.mask_buffer, self.shadow_buffer = pickle.load(open(cache_file, 'rb'))


    def __len__(self):
        return len(self.keys) * self.shadow_per_epoch


    def __getitem__(self, idx):
        """ Note, return format is fixed:
        """
        idx = idx % len(self.keys)

        key = self.keys[idx]

        assert key in self.mask_ds.keys(), 'key {} does not exist in mask_ds'.format(key)
        assert key in self.base_ds.keys(), 'key {} does not exist in base_ds'.format(key)

        # mask = self.mask_ds[key][...]
        # base = self.base_ds[key][...]

        mask = self.mask_buffer[key]
        base = self.shadow_buffer[key]

        # render the shadow
        shadow, ibl = self.render_new_shadow(base)

        # To Tensor
        mask   = torch.Tensor(mask[None, ...])
        ibl    = torch.Tensor(ibl[None, ...])
        shadow = torch.Tensor(shadow[None, ...])

        return {
            'x': {'mask': mask, 'ibl': ibl},
            'y': shadow,
        }


    def prepare_meta(self, hdf5):
        assert 'mask' in hdf5.keys(), 'mask does not exist in hdf5 file'
        assert 'base' in hdf5.keys(), 'base does not exist in hdf5 file'

        keys    = list(hdf5['mask'].keys())
        mask_ds = hdf5['mask']
        base_ds = hdf5['base']

        return keys, mask_ds, base_ds



    def render_new_shadow(self, shadow_bases):
        ih, iw, i, h = shadow_bases.shape

        num = random.randint(0, 50)
        pattern_img = self.random_pattern_generator.get_pattern(iw, ih, num=num, size=0.1, mitsuba=False, dataset=True)

        #  import pdb; pdb.set_trace()

        shadow      = np.tensordot(shadow_bases, pattern_img, axes=([0,1], [0, 1]))
        pattern_img = cv2.resize(pattern_img, (32,16))

        return shadow, pattern_img


    def normalize_energy(self, ibl, energy=30.0):
        if np.sum(ibl) < 1e-3:
            return ibl
        return ibl * energy / np.sum(ibl)


class random_pattern():
    def __init__(self, maximum_blob=50):
        pass

    def y_transform(self, y):
        # y = []
        pass

    def get_pattern(self, w, h, x_density=512, y_density=128, num=50, scale=3.0, size=0.1, energy=3500, mitsuba=False, dataset=True):
        seed = int(time())
        # seed = 19920208

        if num == 0:
            ibl = np.zeros((y_density,x_density))
        else:
            y_fact = y_density/256

            gs = ig.Composite(operator=np.add,
                            generators=[ig.Gaussian(
                                        size=size*ng.UniformRandom(seed=seed+i+4),
                                        scale=scale*(ng.UniformRandom(seed=seed+i+5)+1e-3),
                                        x=ng.UniformRandom(seed=seed+i+1)-0.5,
                                        y=((1.0-ng.UniformRandom(seed=seed+i+2) * y_fact) - 0.5),
                                        aspect_ratio=0.7,
                                        orientation=np.pi*ng.UniformRandom(seed=seed+i+3),
                                        ) for i in range(num)],
                                position=(0, 0),
                                xdensity=512)

            ibl = gs()[:y_density, :]

        # prepare to fix energy inconsistent
        if dataset:
            ibl = self.to_dataset(ibl, w, h)

        if mitsuba:
            return ibl, self.to_mts_ibl(np.copy(ibl))
        else:
            return ibl


    def to_mts_ibl(self, ibl):
        """ Input: 256 x 512 pattern generated ibl
            Output: the ibl in mitsuba ibl
        """
        return np.repeat(ibl[:,:,np.newaxis], 3, axis=2)

    def normalize(self, ibl, energy=30.0):
        total_energy = np.sum(ibl)
        if total_energy < 1e-3:
            # print('small energy: ', total_energy)
            h,w = ibl.shape
            return np.zeros((h,w))

        return ibl * energy / total_energy

    def to_dataset(self, ibl, w, h):
        return self.normalize(cv2.flip(cv2.resize(ibl, (w, h)), 0), 30)



if __name__ == '__main__':
    def test_dataset(ds):
        for i, d in enumerate(tqdm(ds, total=len(ds), desc='Sanity check')):
            x = d['x']
            y = d['y']

            mask = x['mask']
            ibl  = x['ibl']

            assert mask.shape == (1, 256, 256), 'mask shape is wrong: {}'.format(mask.shape)
            assert ibl.shape == (1, 16, 32), 'ibl shape is wrong: {}'.format(ibl.shape)
            assert y.shape == (1, 256, 256), 'y shape is wrong: {}'.format(y.shape)


    opt = {
        'hdf5_file': 'Dataset/SSN/ssn_shadow/shadow_base/ssn_base.hdf5',
        'shadow_per_epoch': 10
    }

    train_ds = SSN_Dataset(opt, True)
    test_ds = SSN_Dataset(opt, False)

    print('Train/Valid: {}/{}'.format(len(train_ds), len(test_ds)))

    test_dataset(train_ds)
    test_dataset(test_ds)
