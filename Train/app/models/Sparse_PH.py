import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torchvision.transforms import Resize
from collections import OrderedDict
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from torchvision.transforms import InterpolationMode


from .abs_model import abs_model
from .blocks import *
from .SSN import SSN
from .SSN_v1 import SSN_v1
from .Loss.Loss import norm_loss, grad_loss
from .Attention_Unet import Attention_Unet 

class Sparse_PH(abs_model):
    def __init__(self, opt):
        mid_act      = opt['model']['mid_act']
        out_act      = opt['model']['out_act']
        in_channels  = opt['model']['in_channels']
        out_channels = opt['model']['out_channels']
        resnet       = opt['model']['resnet']
        backbone     = opt['model']['backbone']

        self.ncols   = opt['hyper_params']['n_cols']
        self.focal   = opt['model']['focal']
        self.clip    = opt['model']['clip']

        self.norm_loss_  = opt['model']['norm_loss']
        self.grad_loss_  = opt['model']['grad_loss']
        self.ggrad_loss_ = opt['model']['ggrad_loss']
        self.lap_loss    = opt['model']['lap_loss']

        self.clip_range = opt['dataset']['linear_scale'] + opt['dataset']['linear_offset']

        if backbone == 'Default':
            self.model = SSN_v1(in_channels=in_channels,
                                out_channels=out_channels,
                                mid_act=mid_act,
                                out_act=out_act,
                                resnet=resnet)
        elif backbone == 'ATTN':
            self.model = Attention_Unet(in_channels, out_channels, mid_act=mid_act, out_act=out_act)

        self.optimizer = get_optimizer(opt, self.model)
        self.visualization = {}

        self.norm_loss = norm_loss()
        self.grad_loss = grad_loss()


    def setup_input(self, x):
        return x


    def forward(self, x):
        return self.model(x)


    def compute_loss(self, y, pred):
        b = y.shape[0]

        # total_loss = avg_norm_loss(y, pred)
        nloss   = self.norm_loss.loss(y, pred) * self.norm_loss_
        gloss   = self.grad_loss.loss(pred) * self.grad_loss_
        ggloss  = self.grad_loss.gloss(y, pred) * self.ggrad_loss_
        laploss = self.grad_loss.laploss(pred) * self.lap_loss

        total_loss = nloss + gloss + ggloss + laploss

        self.loss_log = {
            'norm_loss': nloss.item(),
            'grad_loss': gloss.item(),
            'grad_l1_loss': ggloss.item(),
            'lap_loss': laploss.item(),
        }


        if self.focal:
            total_loss = torch.pow(total_loss, 3)

        return total_loss


    def supervise(self, input_x, y, is_training:bool)->float:
        optimizer = self.optimizer
        model = self.model

        x = input_x['x']

        optimizer.zero_grad()
        pred = self.forward(x)
        if self.clip:
            pred = torch.clip(pred, 0.0, self.clip_range)

        loss = self.compute_loss(y, pred)
        if is_training:
            loss.backward()
            optimizer.step()

        xc = x.shape[1]
        for i in range(xc):
            self.visualization['x{}'.format(i)] = x[:, i:i+1].detach()

        self.visualization['y_fore']    = y[:, 0:1].detach()
        self.visualization['y_back']    = y[:, 1:2].detach()
        self.visualization['pred_fore'] = pred[:, 0:1].detach()
        self.visualization['pred_back'] = pred[:, 1:2].detach()

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
            ret_vis[k] = np.clip(utils.make_grid(plot_v.cpu(), nrow=nrows).numpy().transpose(1,2,0), 0.0, 1.0)
            ret_vis[k] = self.plasma(ret_vis[k])

        return ret_vis


    def get_logs(self):
        return self.loss_log


    def inference(self, x):
        x, device = x['x'], x['device']
        x = torch.from_numpy(x.transpose((2,0,1))).unsqueeze(dim=0).float().to(device)
        pred = self.forward(x)

        pred = pred[0].detach().cpu().numpy().transpose((1,2,0))

        return pred


    def batch_inference(self, x):
        x = x['x']
        pred = self.forward(x)
        return pred


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
    def plasma(self, x):
        norm   = mpl.colors.Normalize(vmin=0.0, vmax=1)
        mapper = cm.ScalarMappable(norm=norm, cmap='plasma')
        bimg   = mapper.to_rgba(x[:,:,0])[:,:,:3]

        return bimg
