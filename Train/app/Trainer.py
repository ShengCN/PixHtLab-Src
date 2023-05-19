import os
from os.path import join
import logging
from collections import OrderedDict
from glob import glob
from enum import Enum

from tqdm import tqdm
from time import time
import torch
import torch.nn as nn

import models
import datasets

from utils import utils
from utils import vis_writer

class Train_State(Enum):
    Train = 0
    Eval  = 1
    Test  = 2


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.setup()


    def setup(self):
        """ Setup Training settings:
            1. Dataloader
            2. Model
            3. Hyper-Params
        """
        opt             = self.opt
        exp_name        = opt['exp_name']
        self.vis_iter   = opt['hyper_params']['vis_iter']
        self.save_iter  = opt['hyper_params']['save_iter']
        self.log_folder = join(opt['hyper_params']['default_folder'], exp_name)
        self.has_testing = 'test_dataset' in opt.keys()
        self.cur_epoch  = 0

        utils.logging_init(opt['exp_name'])
        os.makedirs(self.log_folder, exist_ok=True)

        if not torch.cuda.is_available():
            logging.warn('Not GPU found! Use cpu')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        """ Prepare Dataloader """
        dataloaders           = datasets.create_dataset(opt)
        self.train_dataloader = dataloaders['train']
        self.eval_dataloader  = dataloaders['eval']
        if self.has_testing:
            self.test_dataloader = dataloaders['test']

        """ Prepare Model """
        self.model = models.create_model(opt)

        """ Prepare Hyper Params """
        self.hist_loss = {'epoch_{}_loss'.format(Train_State.Train.name): [],
                          'epoch_{}_loss'.format(Train_State.Eval.name): [],
                          'epoch_{}_loss'.format(Train_State.Test.name): [],
                          'iter_{}_loss'.format(Train_State.Train.name): [],
                          'iter_{}_loss'.format(Train_State.Eval.name): [],
                          'iter_{}_loss'.format(Train_State.Test.name): []}

        """ Prepare Visualizer's writer """
        self.exp_logger = vis_writer.vis_writer(opt)

        """ Logging All Params """
        self.log_all_params()

        """ Setup Training """
        self.setup_training()


    def setup_training(self):
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

        if resume:
            self.resume(weight_file)

        model.set_models(models)
        self.model = model


    def fit(self, dataloader, train_state: Train_State):
        """ Fittin the current dataset
        """
        vis_iter  = self.vis_iter
        save_iter = self.save_iter
        model     = self.model
        epoch     = self.cur_epoch
        models    = model.get_models()

        if train_state == Train_State.Train:
            is_training = True
            torch.set_grad_enabled(True)
        else:
            is_training = False
            torch.set_grad_enabled(False)


        desc = train_state.name

        for k, m in models.items():
            if is_training:
                m.train()
            else:
                m.eval()

        # begin fitting
        epoch_loss = 0.0

        pbar = tqdm(dataloader, total=len(dataloader), desc=desc)
        for i, data in enumerate(pbar):
            x = {k: v.to(self.device) for k, v in data['x'].items()}
            y = data['y'].to(self.device)

            train_x = model.setup_input(x)
            loss    = model.supervise(train_x, y, is_training)

            # record loss
            epoch_loss += loss
            self.add_hist_iter_loss(loss, train_state)

            cur_desc = '{}, iteration Loss: {:.5f}'.format(desc, loss)
            pbar.set_description(cur_desc)

            # record model's losses
            logs = model.get_logs()
            self.logging(logs, train_state)

            # visualization
            if i % vis_iter == 0:
                vis_imgs = model.get_visualize()
                self.visualize(vis_imgs, train_state)

            # we save some visualization results into a html file
            if i % save_iter == 0:
                vis_imgs = model.get_visualize()
                self.save_visualize(vis_imgs, epoch, i, train_state)

        # plot the epoch loss
        epoch_loss = epoch_loss / len(dataloader)
        self.add_hist_epoch_loss(epoch_loss, train_state)
        return epoch_loss


    def train(self):
        """  Training the model
                For i in epochs:
                data = ?
                pred = model.forward(x)
                loss = model.compute_loss(y, pred)
                model.optimize(loss)
        """
        exp_name         = self.opt['exp_name']
        hyper_params     = self.opt['hyper_params']
        train_dataloader = self.train_dataloader
        eval_dataloader  = self.eval_dataloader
        start_epoch      = self.cur_epoch
        has_testing      = self.has_testing

        epochs     = hyper_params['epochs']
        save_epoch = hyper_params['save_epoch']
        desc       = 'Exp. {}'.format(exp_name)

        pbar = tqdm(range(start_epoch, epochs), desc=desc)
        for epoch in pbar:
            train_epoch_loss = self.fit(train_dataloader, train_state=Train_State.Train)
            eval_epoch_loss  = self.fit(eval_dataloader, train_state=Train_State.Eval)

            if has_testing:
                test_epoch_loss  = self.fit(self.test_dataloader, train_state=Train_State.Test)
                # plotting epoch loss
                desc = 'Exp. {}, Epoch: {}, Train loss: {}, Eval loss: {},  Test loss: {}'.format(exp_name,
                                                                                                 epoch,
                                                                                                 train_epoch_loss,
                                                                                                 eval_epoch_loss,
                                                                                                 test_epoch_loss)
                # plotting epoch loss together
                self.exp_logger.plot_losses({'train': train_epoch_loss, 'eval': eval_epoch_loss, 'test': test_epoch_loss}, 'All')
            else:

                # plotting epoch loss
                desc = 'Exp. {}, Epoch: {}, Train loss: {}, Eval loss: {}'.format(exp_name,
                                                                                  epoch,
                                                                                  train_epoch_loss,
                                                                                  eval_epoch_loss)
                # plotting epoch loss together
                self.exp_logger.plot_losses({'train': train_epoch_loss, 'eval': eval_epoch_loss}, 'All')

            # save model
            if epoch % save_epoch == 0:
                self.save(epoch)


            pbar.set_description(desc)
            self.cur_epoch = epoch + 1


    def log_all_params(self):
        opt      = self.opt

        logging.info('')
        logging.info('#' * 60)
        logging.info('Training params:')
        logging.info('Model:')
        logging.info('{}'.format(str(opt['model'])))
        logging.info('-' * 60)

        logging.info('Model Architecture:')
        model = self.model.get_models()
        logging.info('{}'.format(str(model)))
        logging.info('-' * 60)

        logging.info('Dataset:')
        logging.info('{}'.format(str(opt['dataset'])))
        logging.info('-' * 60)

        logging.info('Hyper Params')
        logging.info('{}'.format(str(opt['hyper_params'])))
        logging.info('-' * 60)

        logging.info('Model size')
        logging.info('{} MB'.format(self.get_model_size()))
        logging.info('-' * 60)
        logging.info('#' * 60)
        logging.info('')


    def add_hist_epoch_loss(self, loss, train_state:Train_State):
        key = 'epoch_{}_loss'.format(train_state.name)

        self.hist_loss[key].append(loss)
        self.exp_logger.plot_loss(loss, key)


    def add_hist_iter_loss(self, loss, train_state:Train_State):
        key = 'iter_{}_loss'.format(train_state.name)
        self.hist_loss[key].append(loss)
        self.exp_logger.plot_loss(loss, key)


    def save(self, epoch):
        """ Note, we only save:
               - Models
               - Optimizers
               - history loss
        """
        opt        = self.opt
        gpus       = opt['hyper_params']['gpus']
        model      = self.model
        hist_loss  = self.hist_loss
        log_folder = self.log_folder

        ofile_name = join(log_folder, '{:010d}.pt'.format(epoch))

        tmp_model      = model.get_models()
        tmp_optimizers = model.get_optimizers()

        if len(gpus) > 1:
            for k, v in tmp_model.items():
                tmp_model[k] = v.module

        save_dict = {k:v.state_dict() for k, v in tmp_model.items()}
        for k, opt in tmp_optimizers.items():
            save_dict[k] = opt.state_dict()

        save_dict['hist_loss'] = hist_loss
        save_dict['cur_epoch'] = epoch
        torch.save(save_dict, ofile_name)


    def resume(self, weight_file):
        """ Resume from file
                - Models
                - Optimizers
                - History Loss
        """
        model      = self.model
        models     = self.model.get_models()
        optimizers = self.model.get_optimizers()
        hist_loss  = self.hist_loss
        cur_epoch  = self.cur_epoch
        device     = self.device

        log_folder = self.log_folder

        if weight_file  == 'latest':
            files  = glob(join(log_folder, '*.pt'))

            if len(files) == 0:
                err = 'There is no *.pt file in {}'.format(log_folder)
                logging.error(err)
                raise ValueError(err)

            files.sort()

            weight_file = files[-1]
            logging.info('Resume from file {}'.format(weight_file))
        else:
            weight_file = join(log_folder, weight_file)

            if not os.path.exists(weight_file):
                err = 'There is no {} file'.format(weight_file)
                logging.error(err)
                raise ValueError(err)

        checkpoint = torch.load(weight_file, map_location=device)

        for k, m in models.items():
            models[k].load_state_dict(checkpoint[k])

        for k, o in optimizers.items():
            optimizers[k].load_state_dict(checkpoint[k])

        hist_loss  = checkpoint['hist_loss']
        for k, v in hist_loss.items():
            for l in v:
                self.exp_logger.plot_loss(l, k)

        cur_epoch  = checkpoint['cur_epoch']
        model.set_models(models)
        model.set_optimizers(optimizers)

        self.model     = model
        self.hist_loss = hist_loss
        self.cur_epoch = cur_epoch


    def visualize(self, vis_imgs: OrderedDict, train_state: Train_State):
        prefix = train_state.name

        counter = 0
        for k, v in vis_imgs.items():
            name = '{}/{:02d}_{}'.format(prefix, counter, k)
            self.exp_logger.plot_img(v, name)

            counter += 1


    def logging(self, logs, train_state: Train_State):
        if logs is None:
            return

        prefix = train_state.name
        for k, v in logs.items():
            name = '{}/{}'.format(prefix, k)
            self.exp_logger.plot_loss(v, name)
            # logging.info('{}: {}'.format(name, v))



    def save_visualize(self, vis_imgs: OrderedDict, epoch, iteration, train_state:Train_State):
        label = '{}_{:010d}_{:010d}'.format(train_state.name, epoch, iteration)
        self.exp_logger.save_visualize(vis_imgs, label)


    def get_model_size(self):
        model = self.model

        total_size = 0
        models = model.get_models()
        for k, m in models.items():
            total_size += utils.get_model_size(m)

        # return model size in mb
        return total_size/(1024 ** 2)


if __name__ == '__main__':
    opt = utils.parse_configs()

    trainer = Trainer(opt)
    trainer.train()
