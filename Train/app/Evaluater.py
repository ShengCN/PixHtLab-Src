import os
from os.path import join
import logging
from collections import OrderedDict
from glob import glob
from enum import Enum
import pandas as pd

import numpy as np
from tqdm import tqdm
from time import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import models
import datasets

from utils import utils
from utils import vis_writer
from metrics import metric
import matplotlib

class Evaluater:
    def __init__(self, opt):
        self.opt = opt
        self.setup()


    def setup(self):
        """ Setup Training settings:
            1. Dataloader
            2. Model
            3. Hyper-Params
        """
        opt                = self.opt
        exp_name           = opt['exp_name']
        self.log_folder    = join('Eval', exp_name)
        self.weight_folder = join(opt['hyper_params']['default_folder'], exp_name)
        self.eval_save     = opt['hyper_params']['eval_save']

        os.makedirs(self.log_folder, exist_ok=True)
        utils.logging_init('info', self.log_folder, style='light')

        if not torch.cuda.is_available():
            logging.warn('Not GPU found! Use cpu')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        """ Prepare Dataloader """
        dataloaders           = datasets.create_dataset(opt, True)
        self.eval_dataloader  = dataloaders['eval']
        self.test_dataloader  = None
        if 'test' in dataloaders:
            self.test_dataloader  = dataloaders['test']


        """ Prepare Model """
        self.model = models.create_model(opt)

        self.setup_eval()
        self.log_all_params()


    def setup_eval(self):
        """ Setup states before training
              - Resume?
                - resume model, optimzer's states
                - resume history loss
              - Which GPU?
        """
        opt         = self.opt
        model       = self.model
        resume      = opt['hyper_params']['resume']
        weight_file = opt['hyper_params']['weight_file']
        devices     = opt['hyper_params']['gpus']

        models     = model.get_models()
        if torch.cuda.is_available():
            for k, m in models.items():
                if len(devices) > 1: # mutliple GPU
                    logging.info('Use GPU: {}'.format(','.join([str(d) for d in devices])))
                    models[k] = nn.DataParallel(m, device_ids=devices)
                models[k].to(self.device)

        self.resume(weight_file)
        model.set_models(models)
        self.model = model


    def run_eval(self, dataloader, desc):
        """ Fittin the current dataset
        """
        model        = self.model
        log_folder   = self.log_folder
        eval_ofolder = join(log_folder, desc)
        csv_file     = join(log_folder, '{}_result.csv'.format(desc))

        os.makedirs(eval_ofolder, exist_ok=True)

        models = model.get_models()
        torch.set_grad_enabled(False)
        for k, m in models.items():
            m.eval()


        # begin fitting
        metric_result = []
        for i, data in enumerate(tqdm(dataloader, total=len(dataloader), desc=desc)):
            x = {k: v.to(self.device) for k, v in data['x'].items()}
            y = data['y'].to(self.device)

            train_x = model.setup_input(x)
            pred    = model.batch_inference(train_x)

            b, c, h, w = pred.shape
            # compute metric, save prediction
            for bi in range(b):
                pred_ = pred[bi:bi+1]
                y_    = y[bi:bi+1]

                l2    = metric.norm_metric(pred_, y_)
                rmse  = metric.RMSE_metric(pred_, y_)
                rmses = metric.RMSE_S_metric(pred_, y_)
                ssim  = metric.ssim_metric(pred_, y_)
                psnr  = metric.PSNR_metric(pred_, y_)
                zncc  = metric.ZNCC(pred_, y_)

                if not self.eval_save:
                    cur_metric_result = {'l2': l2.item(),
                                         'ssim': ssim.item(),
                                         'psnr': psnr.item(),
                                         'rmse': rmse.item(),
                                         'rmse_s': rmses.item(),
                                         'zncc': zncc.item(),
                                         }
                    metric_result.append(cur_metric_result)
                    continue

                # save prediction and final results
                prefix      = '{:09d}'.format(i * b + bi)
                gt_output   = join(eval_ofolder, '{}_gt.png'.format(prefix))
                pred_output = join(eval_ofolder, '{}_pred.png'.format(prefix))

                gt_np   = y_[0].detach().cpu().numpy().transpose((1,2,0))
                pred_np = pred_[0].detach().cpu().numpy().transpose((1,2,0))

                self.save_result(gt_output, gt_np)
                self.save_result(pred_output, pred_np)

                matplotlib.use('tkagg')
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(gt_np)
                plt.title('gt')
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.imshow(pred_np)
                plt.title('pred')
                plt.axis('off')
                plt.savefig(join(eval_ofolder, '{}_comp.png'.format(prefix)))
                plt.close()

                metric_result.append({'gt': os.path.relpath(gt_output, log_folder),
                                      'pred': os.path.relpath(pred_output, log_folder),
                                      'comp': os.path.relpath(pred_output, log_folder),
                                      'l2': l2.item(),
                                      'ssim': ssim.item(),
                                      'psnr': psnr.item(),
                                      'rmse': rmse.item(),
                                      'rmse_s': rmses.item(),
                                      'zncc': zncc.item()})

        df = pd.DataFrame(data=metric_result)
        df.to_csv(csv_file, index=False)

        # compute avg
        l2    = df['l2'].mean()
        rmse  = df['rmse'].mean()
        rmses = df['rmse_s'].mean()
        ssim  = df['ssim'].mean()
        psnr  = df['psnr'].mean()
        zncc  = df['zncc'].mean()
        return {'l2': l2, 'rmse': rmse, 'rmse_s':rmses, 'ssim': ssim, 'psnr': psnr, 'zncc': zncc}


    def eval(self):
        exp_name         = self.opt['exp_name']
        hyper_params     = self.opt['hyper_params']
        weight_file      = hyper_params['weight_file']
        eval_dataloader  = self.eval_dataloader
        test_dataloader  = self.test_dataloader

        eval_result = self.run_eval(eval_dataloader, desc='eval')
        logging.info('')
        logging.info('=' * 100)
        logging.info('Exp {}, weight: {}'.format(exp_name, weight_file))
        logging.info('Eval. L2: {}, RMSE: {}, RMSE_S: {} SSIM: {}, PSNR: {}, ZNCC: {}'.format(
            eval_result['l2'],
            eval_result['rmse'],
            eval_result['rmse_s'],
            eval_result['ssim'],
            eval_result['psnr'],
            eval_result['zncc']))

        if test_dataloader is not None:
            test_result = self.run_eval(test_dataloader, desc='test')
            logging.info('Exp {}, weight: {}'.format(exp_name, weight_file))
            logging.info('Test. L2: {}, RMSE: {}, RMSE_S: {} SSIM: {}, PSNR: {}, ZNCC: {}'.format(
                test_result['l2'],
                test_result['rmse'],
                test_result['rmse_s'],
                test_result['ssim'],
                test_result['psnr'],
                test_result['zncc']))

        logging.info('=' * 100)
        logging.info('')

        # write to a file to write latex for convenience
        latex_output = join(os.path.dirname(self.log_folder), 'latex.txt')
        with open(latex_output, 'a') as f:
            f.write('{} & {:.4f}  & {:.4f} & {:.4f} & {:.4f} \n'.format(
                exp_name, 
                test_result['rmse'],
                test_result['rmse_s'],
                test_result['ssim'],
                test_result['zncc']))



    def log_all_params(self):
        opt      = self.opt

        logging.info('')
        logging.info('-' * 60)
        logging.info('Training params:')
        logging.info('Model:')
        self.log_dict(opt['model'])
        logging.info('-' * 60)

        logging.info('Dataset:')
        self.log_dict(opt['dataset'])
        logging.info('-' * 60)

        logging.info('Hyper Params')
        self.log_dict(opt['hyper_params'])
        logging.info('Model size')
        logging.info('{} MB'.format(self.get_model_size()))
        logging.info('-' * 60)
        logging.info('')


    def log_dict(self, info:dict):
        for k, v in info.items():
            logging.info('{}: {}'.format(k, v))



    def resume(self, weight_file):
        """ Resume from file
                - Models
                - Optimizers
                - History Loss
        """
        model         = self.model
        models        = self.model.get_models()
        device        = self.device
        weight_folder = self.weight_folder

        if weight_file  == 'latest':
            files  = glob(join(weight_folder, '*.pt'))

            if len(files) == 0:
                err = 'There is no *.pt file in {}'.format(weight_folder)
                logging.error(err)
                raise ValueError(err)

            files.sort()

            weight_file = files[-1]
            logging.info('Resume from file {}'.format(weight_file))
        else:
            weight_file = join(weight_folder, weight_file)

            if not os.path.exists(weight_file):
                err = 'There is no {} file'.format(weight_file)
                logging.error(err)
                raise ValueError(err)

        checkpoint = torch.load(weight_file, map_location=device)

        for k, m in models.items():
            models[k].load_state_dict(checkpoint[k])
        model.set_models(models)

        self.model     = model


    def get_model_size(self):
        model = self.model

        total_size = 0
        models = model.get_models()
        for k, m in models.items():
            total_size += utils.get_model_size(m)

        # return model size in mb
        return total_size/(1024 ** 2)


    def save_result(self, ofile, img_np):
        h, w, c = img_np.shape
        if c == 1:
            img_np = np.repeat(img_np, 3, axis=2)

        img_np = np.clip(img_np, 0.0, 1.0)
        plt.imsave(ofile, img_np)



if __name__ == '__main__':
    opt = utils.parse_configs()

    evaluater = Evaluater(opt)
    evaluater.eval()
