import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from collections import OrderedDict

from .abs_model import abs_model
from .blocks import *
from .Loss.Loss import avg_norm_loss

class Template(abs_model):
    """ Standard Unet Implementation
        src: https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, opt):
        resunet      = opt['model']['resunet']
        out_act      = opt['model']['out_act']
        norm_type    = opt['model']['norm_type']
        in_channels  = opt['model']['in_channels']
        out_channels = opt['model']['out_channels']
        self.ncols   = opt['hyper_params']['n_cols']

        self.model = Unet(in_channels=in_channels,
                          out_channels=out_channels,
                          norm_type=norm_type,
                          out_act=out_act,
                          resunet=resunet)

        self.optimizer = get_optimizer(opt, self.model)
        self.visualization = {}


    def setup_input(self, x):
        return x


    def forward(self, x):
        return self.model(x)


    def compute_loss(self, y, pred):
        return avg_norm_loss(y, pred)


    def supervise(self, input_x, y, is_training:bool)->float:
        optimizer = self.optimizer
        model = self.model

        optimizer.zero_grad()
        pred = model(input_x)
        loss = self.compute_loss(y, pred)

        if is_training:
            loss.backward()
            optimizer.step()

        self.visualization['y']    = pred.detach()
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
            ret_vis[k] = utils.make_grid(plot_v.cpu(), nrow=nrows).numpy().transpose(1,2,0)

        return ret_vis


    def inference(self, x):
        # TODO
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
