import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from collections import OrderedDict
import numpy as np

from .abs_model import abs_model
from .Loss.Loss import norm_loss
from .blocks import *
from .SSN_Model import SSN_Model


class SSN(abs_model):
    def __init__(self, opt):
        mid_act      = opt['model']['mid_act']
        out_act      = opt['model']['out_act']
        in_channels  = opt['model']['in_channels']
        out_channels = opt['model']['out_channels']
        self.ncols   = opt['hyper_params']['n_cols']

        self.model         = SSN_Model(in_channels=in_channels, out_channels=out_channels, mid_act=mid_act, out_act=out_act)
        self.optimizer     = get_optimizer(opt, self.model)
        self.visualization = {}

        self.norm_loss_ = norm_loss(norm=1)

    def setup_input(self, x):
        return x


    def forward(self, x):
        keys = ['mask', 'ibl']

        for k in keys:
            assert k in x.keys(), '{} not in input'.format(k)

        mask = x['mask']
        ibl  = x['ibl']

        return self.model(mask, ibl)


    def compute_loss(self, y, pred):
        total_loss = self.norm_loss_.loss(y, pred)
        return total_loss


    def supervise(self, input_x, y, is_training:bool)->float:
        optimizer = self.optimizer
        model = self.model

        optimizer.zero_grad()
        pred = self.forward(input_x)
        loss = self.compute_loss(y, pred)

        if is_training:
            loss.backward()
            optimizer.step()

        self.visualization['mask'] = input_x['mask'].detach()
        self.visualization['ibl'] = input_x['ibl'].detach()
        self.visualization['y']    = y.detach()
        self.visualization['pred'] = pred.detach()

        return loss.item()


    def get_visualize(self) -> OrderedDict:
        """ Convert to visualization numpy array
        """
        nrows          = self.ncols
        visualizations = self.visualization
        ret_vis        = OrderedDict()

        for k, v in visualizations.items():
            batch = v.shape[0]
            n     = min(nrows, batch)

            plot_v = v[:n]
            plot_v = (plot_v - plot_v.min())/(plot_v.max() - plot_v.min())
            ret_vis[k] = np.clip(utils.make_grid(plot_v.cpu(), nrow=nrows).numpy().transpose(1,2,0), 0.0, 1.0)

        return ret_vis


    def get_logs(self):
        pass


    def inference(self, x):
        pass

    def batch_inference(self, x):
        # TODO
        pass


    """ Getter & Setter
    """
    def get_models(self) -> dict:
        return {'model': self.model}


    def get_optimizers(self) -> dict:
        return {'optimizer': self.optimizer}


    def set_models(self, models: dict) :
        # input test
        if 'model' not in models.keys():
            raise ValueError('{} not in self.model'.format('model'))

        self.model = models['model']


    def set_optimizers(self, optimizer: dict):
        self.optimizer = optimizer['optimizer']


    ####################
    # Personal Methods #
    ####################


