import sys

sys.path.append("..")

import os
from os.path import join
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import random
# import matplotlib.pyplot as plt
import cv2
from params import params
from .random_pattern import random_pattern
from .perturb_touch import random_perturb


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, is_transpose=True):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if is_transpose:
            img = img.transpose((0, 3, 1, 2))
        return torch.Tensor(img)


class SSN_Dataset(Dataset):
    def __init__(self, ds_dir, hd_dir, is_training, fake_batch_size=8):
        start = time.time()
        self.fake_batch_size = fake_batch_size
        # # of samples in each group
        # magic number here
        self.ibl_group_size = 16

        parameter = params().get_params()

        # (shadow_path, mask_path)
        self.meta_data = self.init_meta(ds_dir, hd_dir)

        self.is_training = is_training
        self.to_tensor = ToTensor()

        end = time.time()
        print("Dataset initialize spent: {} ms".format(end - start))

        # fake random
        np.random.seed(19950220)
        np.random.shuffle(self.meta_data)

        self.valid_divide = 10
        if parameter.small_ds:
            self.meta_data = self.meta_data[:len(self.meta_data) // self.valid_divide]

        self.training_num = len(self.meta_data) - len(self.meta_data) // self.valid_divide
        print('training: {}, validation: {}'.format(self.training_num, len(self.meta_data) // self.valid_divide))

        self.random_pattern_generator = random_pattern()

        self.thread_id = os.getpid()
        self.seed = os.getpid()
        self.perturb = not parameter.pred_touch and not parameter.touch_loss
        self.size_interval = 0.5 / 100
        self.ori_interval = np.pi / 100

        self.soft_distribution = [[np.exp(-0.4 * (i - j) ** 2) for i in np.arange(0.5 / self.size_interval)]
                                  for j in np.arange(0.5 / self.size_interval)]

    def __len__(self):
        if self.is_training:
            return self.training_num
        else:
            # return len(self.meta_data) - self.training_num
            return len(self.meta_data) // self.valid_divide

    def __getitem__(self, idx):
        if self.is_training and idx > self.training_num:
            print("error")
        # offset to validation set
        if not self.is_training:
            idx = self.training_num + idx

        cur_seed = idx * 1234 + os.getpid() + time.time()
        random.seed(cur_seed)

        # random ibls
        shadow_path, mask_path, hard_path, touch_path = self.meta_data[idx]
        hard_folder = hard_path.replace(hard_path.split('/')[-1], '')
        if os.path.exists(hard_folder):
            # hard_shadow = cv2.imread(hard_folder)

            mask_img = cv2.imread(mask_path)
            mask_img = mask_img[:, :, 0]
            if mask_img.dtype == np.uint8:
                mask_img = mask_img / 255.0
            mask_img, shadow_bases = np.expand_dims(mask_img, axis=2), np.load(shadow_path)

            w, h, c, m = shadow_bases.shape
            shadow_soft_list = []
            shadow_hard_list = []
            size_list = []
            orientation_list = []
            mask_img_list = []
            for i in range(int(self.fake_batch_size)):
                shadow_img, light_img, size, orientation = self.render_new_shadow(shadow_bases)

                h, w = mask_img.shape[0], mask_img.shape[1]
                hi, wi = np.where(light_img == light_img.max())

                while len(hi) > 1:
                    shadow_img, light_img, size, orientation = self.render_new_shadow(shadow_bases)
                    hi, wi = np.where(light_img == light_img.max())
                size_soft = np.array(self.soft_distribution[int(size / self.size_interval)])
                ori_soft = np.array(self.soft_distribution[int(orientation / self.ori_interval)])
                prefix = '_ibli_' + str(int(wi * 8)) + '_iblj_' + str(int(hi * 8) + 128) + '_shadow.png'
                shadow_hard_path = hard_path.replace('_shadow.png', prefix)
                shadow_base = cv2.imread(shadow_hard_path, -1)[:, :, 0] / 255.0
                shadow_base = np.expand_dims(shadow_base, axis=2)
                shadow_base = self.line_aug(shadow_base)
                shadow_soft_list.append(shadow_img)
                shadow_hard_list.append(shadow_base)
                size_list.append(size_soft)
                orientation_list.append(ori_soft)
                mask_img_list.append(mask_img)
            shadow_softs = np.array(shadow_soft_list)
            shadow_hards = np.array(shadow_hard_list)
            sizes = np.array(size_list)
            orientations = np.array(orientation_list)
            mask_imgs = np.array(mask_img_list)

            # touch_img = self.read_img(touch_path)
            # touch_img = touch_img[:, :, 0:1]

            #         if self.perturb:
            #             touch_img = random_perturb(touch_img)

            # input_img = np.concatenate((mask_img, touch_img), axis=2)
            size = torch.Tensor(sizes)
            ori = torch.Tensor(orientations)
            hard_shadow, soft_shadow, mask_img = self.to_tensor(shadow_hards), self.to_tensor(
                shadow_softs), self.to_tensor(
                mask_imgs)
            return {"hard_shadow": hard_shadow, "soft_shadow": soft_shadow, "mask_img": mask_img, "size": size,
                    "angle": ori}
        else:
            mask_img = cv2.imread(mask_path)
            mask_img = mask_img[:, :, 0]
            if mask_img.dtype == np.uint8:
                mask_img = mask_img / 255.0
            mask_img, shadow_bases = np.expand_dims(mask_img, axis=2), 1.0 - np.load(shadow_path)

            w, h, c, m = shadow_bases.shape
            shadow_soft_list = []
            shadow_hard_list = []
            size_list = []
            orientation_list = []
            mask_img_list = []
            for i in range(int(self.fake_batch_size)):
                shadow_img, light_img, size, orientation = self.render_new_shadow(shadow_bases)

                h, w = mask_img.shape[0], mask_img.shape[1]
                hi, wi = np.where(light_img == light_img.max())

                while len(hi) > 1:
                    shadow_img, light_img, size, orientation = self.render_new_shadow(shadow_bases)
                    hi, wi, _ = np.where(light_img == light_img[:, :, :].max())
                size_soft = np.array(self.soft_distribution[int(size / self.size_interval)])
                ori_soft = np.array(self.soft_distribution[int(orientation / self.ori_interval)])

                shadow_base = shadow_bases[:, :, wi, hi]
                shadow_base[shadow_base > 0.3] = 1
                shadow_base[shadow_base < 0.4] = 0
                shadow_base = self.line_aug(shadow_base)
                mask_img = np.expand_dims(cv2.resize(mask_img, (512, 512)), axis=2)
                shadow_base = np.expand_dims(cv2.resize(shadow_base, (512, 512)), axis=2)
                shadow_img = np.expand_dims(cv2.resize(shadow_img, (512, 512)), axis=2)
                shadow_soft_list.append(shadow_img)
                shadow_hard_list.append(shadow_base)
                size_list.append(size_soft)
                orientation_list.append(ori_soft)
                mask_img_list.append(mask_img)
            shadow_softs = np.array(shadow_soft_list)
            shadow_hards = np.array(shadow_hard_list)
            sizes = np.array(size_list)
            orientations = np.array(orientation_list)
            mask_imgs = np.array(mask_img_list)

            # touch_img = self.read_img(touch_path)
            # touch_img = touch_img[:, :, 0:1]

            #         if self.perturb:
            #             touch_img = random_perturb(touch_img)

            # input_img = np.concatenate((mask_img, touch_img), axis=2)
            size = torch.Tensor(sizes)
            ori = torch.Tensor(orientations)

            hard_shadow, soft_shadow, mask_img = self.to_tensor(shadow_hards), self.to_tensor(
                shadow_softs), self.to_tensor(
                mask_imgs)

            return {"hard_shadow": hard_shadow, "soft_shadow": soft_shadow, "mask_img": mask_img, "size": size,
                    "angle": ori}

    def init_meta(self, ds_dir, hd_dir):
        metadata = []
        # base_folder = join(ds_dir, 'base')
        # mask_folder = join(ds_dir, 'mask')
        # hard_folder = join(ds_dir, 'hard')
        # touch_folder = join(ds_dir, 'touch')
        # model_list = [f for f in os.listdir(base_folder) if os.path.isdir(join(base_folder, f))]
        # for m in model_list:
        #     shadow_folder, cur_mask_folder = join(base_folder, m), join(mask_folder, m)
        #     shadows = [f for f in os.listdir(shadow_folder) if f.find('_shadow.npy') != -1]
        #     for s in shadows:
        #         prefix = s[:s.find('_shadow')]
        #         metadata.append((join(shadow_folder, s),
        #                          join(cur_mask_folder, prefix + '_mask.png'),
        #                          join(join(hard_folder, m), prefix + '_shadow.png'),
        #                          join(join(touch_folder, m), prefix + '_touch.png')))

        base_folder = join(hd_dir, 'base')
        mask_folder = join(hd_dir, 'mask')
        hard_folder = join(hd_dir, 'hard')
        touch_folder = join(hd_dir, 'touch')
        model_list = [f for f in os.listdir(base_folder) if os.path.isdir(join(base_folder, f))]
        for m in model_list:
            shadow_folder, cur_mask_folder = join(base_folder, m), join(mask_folder, m)
            shadows = [f for f in os.listdir(shadow_folder) if f.find('_shadow.npy') != -1]
            for s in shadows:
                prefix = s[:s.find('_shadow')]
                metadata.append((join(shadow_folder, s),
                                 join(cur_mask_folder, prefix + '_mask.png'),
                                 join(join(hard_folder, m), prefix + '_shadow.png'),
                                 join(join(touch_folder, m), prefix + '_touch.png')))

        return metadata

    def line_aug(self, shadow):
        p = np.random.random()
        if p > 0.6:
            k = np.tan(min((np.random.random() + 0.000000001), 0.999) * np.pi - np.pi / 2)
            x, y, c = shadow.shape
            b_max = y - x * k
            line_num = np.random.randint(1, 20)
            b_list = np.random.random(line_num) * b_max
            x_coord = np.tile(np.arange(shadow.shape[1])[None, :], (shadow.shape[0], 1))
            y_coord = np.tile(np.arange(shadow.shape[0])[:, None], (1, shadow.shape[1]))

            for b in b_list:
                mask_res = y_coord - k * x_coord - b
                shadow[np.abs(mask_res) < 1] = 0
        return shadow

    def get_prefix(self, path):
        folder = os.path.dirname(path)
        basename = os.path.basename(path)
        return os.path.join(folder, basename[:basename.find('_')])

    def render_new_shadow(self, shadow_bases):
        shadow_bases = shadow_bases[:, :, :, :]
        h, w, iw, ih = shadow_bases.shape

        num = random.randint(0, 50)
        pattern_img, size, orientation = self.random_pattern_generator.get_pattern(iw, ih, num=num, size=0.5,
                                                                                   mitsuba=False)

        # flip to mitsuba ibl
        pattern_img = self.normalize_energy(cv2.flip(cv2.resize(pattern_img, (iw, ih)), 0))
        shadow = np.tensordot(shadow_bases, pattern_img, axes=([2, 3], [1, 0]))
        # pattern_img = np.expand_dims(cv2.resize(pattern_img, (iw, 16)), 2)

        return np.expand_dims(shadow, 2), pattern_img, size, orientation

    def get_min_max(self, batch_data, name):
        print('{} min: {}, max: {}'.format(name, np.min(batch_data), np.max(batch_data)))

    def log(self, log_info):
        with open('log.txt', 'a+') as f:
            f.write(log_info)

    def normalize_energy(self, ibl, energy=30.0):
        if np.sum(ibl) < 1e-3:
            return ibl
        return ibl * energy / np.sum(ibl)
