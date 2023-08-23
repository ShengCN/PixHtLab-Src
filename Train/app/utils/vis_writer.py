import os
from os.path import join
from collections import OrderedDict

import pandas as pd
import torch
from torchvision import utils
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from . import utils
from .html_maker import df_html


class vis_writer:
    def __init__(self, opt):
        """
        - checkpoint
        - exp_name
            - 000...00.pt
            - logs/
            - saved
            - imgs/
            - index.html
        """

        exp_name     = opt['exp_name']
        hyper_params = opt['hyper_params']

        cur_ofolder     = join(hyper_params['default_folder'], exp_name)
        log_folder      = join(cur_ofolder, 'logs', utils.get_cur_time_stamp())
        save_folder     = join(cur_ofolder, 'saved')
        save_img_folder = join(save_folder, 'imgs')
        self.exp_name   = exp_name

        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_img_folder, exist_ok=True)

        self.writer        = SummaryWriter(log_folder)
        self.save_folder   = save_folder
        self.iter_counters = {}


    def plot_loss(self, loss:float, name:str):
        writer = self.writer
        iter   = self.update_iter(name)

        writer.add_scalar('Loss/{}'.format(name), loss, iter)


    def plot_losses(self, all_loss: dict, name:str):
        writer = self.writer
        iter = self.update_iter(name)

        writer.add_scalars('Loss/{}'.format(name), all_loss, iter)


    def plot_img(self, img_np, name:str):
        writer = self.writer
        iter = self.update_iter(name)

        h, w, c = img_np.shape
        writer.add_image(name, img_np, iter, dataformats='HWC')


    def update_iter(self, name):
        if name not in self.iter_counters.keys():
            self.iter_counters[name] = 0
        else:
            self.iter_counters[name] += 1

        return self.iter_counters[name]


    def save_visualize(self, vis_images:OrderedDict, label:str):
        """ Save results to a html
        """
        save_folder = self.save_folder
        csv         = join(save_folder, 'meta.csv')
        img_ofolder = join(save_folder, 'imgs')
        html_file   = join(save_folder, 'index.html')
        exp_name    = self.exp_name

        new_data = {}
        for k, v in vis_images.items():
            key = '{}_{}'.format(label, k)
            opath = join(img_ofolder, key + '.png')
            new_data[k] = opath

            if v.max() > 1.0 or v.min() < 0.0:
                err = '{} is out of range'.format(k)
                logging.error(err)
                raise ValueError(err)

            plt.imsave(opath, v)

        if os.path.exists(csv):
            df     = pd.read_csv(csv)
            tmp_df = pd.DataFrame(data=[new_data])
            df     = pd.concat([df, tmp_df], ignore_index=True)
        else:
            df = pd.DataFrame(data=[new_data])

        # make a html
        if os.path.exists(html_file):
            os.remove(html_file)

        try:
            df_html(df, save_folder, self.exp_name)
            df.to_csv(csv, index=False)
        except BaseException as err:
            logging.error(err)
            os.remove(csv)
