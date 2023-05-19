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
import sys
from .camera import axis_camera, xyh2xyz, compute_normal, normalize_vec3


BINs              = 100
MAX_RAD           = 20
size_interval     = MAX_RAD / BINs
soft_distribution = [[np.exp(-0.2 * (i - j) ** 2) for i in np.arange(BINs)] for j in np.arange(BINs)]

CUR_H5_DS      = 'xy_256'
MASK_IND       = 0
HMAP_IND       = 1
SHADOW_IND     = 3
RECHMAP_IND    = 4
NHMAP_DIFF_IND = 7
FHMAP_DIFF_IND = 8
NHMAP_DIS_IND  = 9
FHMAP_DIS_IND  = 10
SHADING_IND    = 12
REC_GRADX_IND  = 15
REC_GRADY_IND  = 16
SS_IND         = 17
SHADOW_BOUNDARY = 18
OUT_SHADOW_VALUE = 2

def get_hdf5_root(is_training):
    if is_training:
        return '{}/{}'.format('train', CUR_H5_DS)
    else:
        return '{}/{}'.format('eval', CUR_H5_DS)


def get_hdf5_light_root(is_training):
    if is_training:
        return '{}/{}'.format('train', 'l')
    else:
        return '{}/{}'.format('eval', 'l')


def get_softness(radius):
    softness = np.array(soft_distribution[int(radius / size_interval)])
    return softness


def hdf52data(ds, idx, input_type, rech_grad=True, ignore_shading=False, is_training=True):
    xy_data      = ds[get_hdf5_root(is_training)][idx]
    light_data   = ds[get_hdf5_light_root(is_training)][idx]

    if rech_grad:
        # grad_factor = 512.0
        grad_factor = 2.0
        rechmap_x = xy_data[REC_GRADX_IND][...,np.newaxis] * grad_factor
        rechmap_y = xy_data[REC_GRADY_IND][...,np.newaxis] * grad_factor
    else:
        rechmap_x = xy_data[RECHMAP_IND][..., np.newaxis]
        rechmap_y = xy_data[RECHMAP_IND][..., np.newaxis]

    if input_type == 'BC_Only_Shadow':
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        shadow          = xy_data[SHADOW_IND][..., None]
        shadow_boundary = xy_data[SHADOW_BOUNDARY][..., None]

        x = np.concatenate([hmap, shadow, shadow_boundary], axis=2)

    elif input_type == 'Buffer_Channel':
        # mask       = xy_data[MASK_IND][...,np.newaxis]
        hmap       = xy_data[HMAP_IND][..., np.newaxis]
        nhmap_diff = xy_data[NHMAP_DIFF_IND][...,np.newaxis]
        nhmap_dis  = xy_data[NHMAP_DIS_IND][...,np.newaxis]
        shadow     = xy_data[SHADOW_IND]

        x = np.concatenate([nhmap_diff,  nhmap_dis], axis = 2)

        # mark out of shadow region  OUT_SHADOW_VALUE
        x[shadow < 1e-1] = OUT_SHADOW_VALUE
        x = np.concatenate([hmap, x, rechmap_x, rechmap_y], axis=2)

    elif input_type == 'BC_Boundary':
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        nhmap_diff      = xy_data[NHMAP_DIFF_IND][...,np.newaxis]
        nhmap_dis       = xy_data[NHMAP_DIS_IND][...,np.newaxis]
        shadow          = xy_data[SHADOW_IND]
        shadow_boundary = xy_data[SHADOW_BOUNDARY][..., None]

        x = np.concatenate([nhmap_diff,  nhmap_dis], axis = 2)

        # mark out of shadow region  OUT_SHADOW_VALUE
        x[shadow < 1e-1] = OUT_SHADOW_VALUE

        x = np.concatenate([hmap, x, rechmap_x, rechmap_y, shadow_boundary], axis=2)

    elif input_type == 'BC_Boundary_XYH_DIS':
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        nhmap_diff      = xy_data[NHMAP_DIFF_IND][...,np.newaxis]
        nhmap_dis       = xy_data[NHMAP_DIS_IND][...,np.newaxis]
        shadow          = xy_data[SHADOW_IND]
        shadow_boundary = xy_data[SHADOW_BOUNDARY][..., None]

        x = np.sqrt(nhmap_diff ** 2 + nhmap_dis ** 2)

        # mark out of shadow region  OUT_SHADOW_VALUE
        x[shadow < 1e-1] = OUT_SHADOW_VALUE

        x = np.concatenate([hmap, x, shadow_boundary, rechmap_x, rechmap_y], axis=2)

    elif input_type == 'SSG_RHMAP':
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        shadow          = xy_data[SHADOW_IND][..., None]

        x = np.concatenate([hmap, shadow, rechmap_x, rechmap_y], axis=2)

    elif input_type == 'SSG_XYHDIS':
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        nhmap_diff      = xy_data[NHMAP_DIFF_IND][...,np.newaxis]
        nhmap_dis       = xy_data[NHMAP_DIS_IND][...,np.newaxis]
        shadow          = xy_data[SHADOW_IND]
        shadow_boundary = xy_data[SHADOW_BOUNDARY][..., None]

        x = np.sqrt(nhmap_diff ** 2 + nhmap_dis ** 2)

        # mark out of shadow region  OUT_SHADOW_VALUE
        x[shadow < 1e-1] = OUT_SHADOW_VALUE

        x = np.concatenate([hmap, x], axis=2)

    elif input_type == 'SSG_XYHDIS_BC':
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        nhmap_diff      = xy_data[NHMAP_DIFF_IND][...,np.newaxis]
        nhmap_dis       = xy_data[NHMAP_DIS_IND][...,np.newaxis]
        shadow          = xy_data[SHADOW_IND]
        shadow_boundary = xy_data[SHADOW_BOUNDARY][..., None]

        x = np.sqrt(nhmap_diff ** 2 + nhmap_dis ** 2)

        # mark out of shadow region  OUT_SHADOW_VALUE
        x[shadow < 1e-1] = OUT_SHADOW_VALUE

        x = np.concatenate([hmap, x, shadow_boundary], axis=2)

    elif input_type == 'SSG_XYHDIS_RHMAP':
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        nhmap_diff      = xy_data[NHMAP_DIFF_IND][...,np.newaxis]
        nhmap_dis       = xy_data[NHMAP_DIS_IND][...,np.newaxis]
        shadow          = xy_data[SHADOW_IND]

        x = np.sqrt(nhmap_diff ** 2 + nhmap_dis ** 2)

        # mark out of shadow region  OUT_SHADOW_VALUE
        x[shadow < 1e-1] = OUT_SHADOW_VALUE

        x = np.concatenate([hmap, x, rechmap_x, rechmap_y], axis=2)
    elif input_type == 'Boundary_RH':
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        shadow          = xy_data[SHADOW_IND][..., None]
        shadow_boundary = xy_data[SHADOW_BOUNDARY][..., None]

        x = np.concatenate([hmap, shadow, rechmap_x, rechmap_y, shadow_boundary], axis=2)

    elif input_type == 'SSG':
        # mask   = xy_data[MASK_IND][...,np.newaxis]
        hmap            = xy_data[HMAP_IND][..., np.newaxis]
        shadow = xy_data[SHADOW_IND][...,np.newaxis]
        x = np.concatenate([hmap, shadow], axis=2)

    elif input_type == 'SSG_NH':
        # mask   = xy_data[MASK_IND][...,np.newaxis]
        hmap   = xy_data[HMAP_IND][..., np.newaxis]
        shadow = xy_data[SHADOW_IND][...,np.newaxis]
        nhmap_diff = xy_data[NHMAP_DIFF_IND][...,np.newaxis]
        # nhmap_dis  = xy_data[NHMAP_DIS_IND][...,np.newaxis]
        x = np.concatenate([hmap, shadow, nhmap_diff], axis=2)

    elif input_type == 'SSG_NH_DIS':
        # mask   = xy_data[MASK_IND][...,np.newaxis]
        hmap   = xy_data[HMAP_IND][..., np.newaxis]
        shadow = xy_data[SHADOW_IND][...,np.newaxis]
        nhmap_diff = xy_data[NHMAP_DIFF_IND][...,np.newaxis]
        nhmap_dis  = xy_data[NHMAP_DIS_IND][...,np.newaxis]
        x = np.concatenate([hmap, shadow, nhmap_diff, nhmap_dis], axis=2)

    elif input_type == 'SSG_H':
        mask   = xy_data[MASK_IND][...,np.newaxis]
        shadow = xy_data[SHADOW_IND][...,np.newaxis]
        x = np.concatenate([mask, shadow, rechmap_x, rechmap_y], axis=2)
    else:
        raise ValueError('Model {} has not implemented yet'.format(input_type))

    y      = xy_data[SS_IND][..., np.newaxis]
    light  = light_data
    radius = light[-1]

    if radius >= MAX_RAD:
        logging.error("radius {} is too big, ignore this input".format(radius))
        return None, None, None, None

    softness = get_softness(radius)

    if ignore_shading:
        return x.astype(np.float32), softness.astype(np.float32), y.astype(np.float32), light
    else:
        shading = xy_data[SHADING_IND:SHADING_IND+3]
        return x.astype(np.float32), softness.astype(np.float32), y.astype(np.float32), light, shading


class GSSN_Dataset(Dataset):
    def __init__(self, opt, is_training):
        random.seed(19920208)
        np.random.seed(19920208)

        self.is_training = is_training
        self.transform = transforms.Compose([ transforms.ToTensor() ])

        self.rech_grad  = opt['rech_grad']
        self.input_type = opt['type']
        try:
            self.df_hdf5    = h5py.File(opt['hdf5_file'], 'r')
        except BaseException as err:
            logging.error('{}: {}'.format(opt['hdf5_file'], err))

        cur_ds = get_hdf5_root(is_training)
        self.total_data_len = self.df_hdf5[cur_ds].shape[0]

        logging.info('Use hdf5. Data: {}'.format(self.total_data_len))


    def __len__(self):
        return self.total_data_len


    def __getitem__(self, idx):
        """ Note, return format is fixed:
        """
        try:
            x, softness, y, light  = hdf52data(self.df_hdf5,
                                               idx,
                                               self.input_type,
                                               rech_grad=self.rech_grad,
                                               ignore_shading=True,
                                               is_training=self.is_training)

            return {
                'x':{'x':self.transform(x).float(), 'softness': softness.astype(np.float32)},
                'y':self.transform(y).float(),
                'light':light}

        except BaseException as err:
            import pdb; pdb.set_trace()
            logging.error('File {} has problem: {}'.format(idx, err))
            self.df_hdf5.close()


if __name__ == '__main__':
    opt = {'type': 'Buffer_Channel',
                    'hdf5_file': 'Dataset1/general_scenes/train/ALL_SIZE_WALL/dataset.hdf5',
                    'rech_grad': False}
    ds = GSSN_Dataset(opt, True)

    # for i, d in enumerate(ds):
    #     x, y = d['x'], d['y']

    diff_range     = [100, -100]
    dist_range     = [100, -100]
    rech_range     = [100, -100]
    softness_range = [100, -100]

    ds_loader = DataLoader(ds, num_workers=32, batch_size=32)
    for i, d in enumerate(tqdm(ds_loader, total=len(ds_loader))):
        x, y = d['x'], d['y']
        x, softness = x['x'], x['softness']

        diff_range[0] = min(x[:,1].min(), diff_range[0])
        diff_range[1] = max(x[:,1].max(), diff_range[1])

        dist_range[0] = min(x[:,2].min(), dist_range[0])
        dist_range[1] = max(x[:,2].max(), dist_range[1])

        rech_range[0] = min(x[:,3].min(), rech_range[0])
        rech_range[1] = max(x[:,3].max(), rech_range[1])

        softness_range[0] = min(softness.min(), softness_range[0])
        softness_range[1] = max(softness.max(), softness_range[1])

    print('diff range: ', diff_range)
    print('dist range: ', dist_range)
    print('rech range: ', rech_range)
    print('softness range: ', softness_range)
