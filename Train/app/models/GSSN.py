import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from collections import OrderedDict
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl

from .abs_model import abs_model
from .blocks import *
from .SSN import SSN
from .SSN_v1 import SSN_v1
from .Loss.Loss import norm_loss


class GSSN(abs_model):
    def __init__(self, opt):
        mid_act      = opt['model']['mid_act']
        out_act      = opt['model']['out_act']
        in_channels  = opt['model']['in_channels']
        out_channels = opt['model']['out_channels']
        resnet       = opt['model']['resnet']
        self.ncols   = opt['hyper_params']['n_cols']
        self.focal   = opt['model']['focal']

        if 'backbone' not in opt['model'].keys():
            self.model = SSN(in_channels=in_channels,
                             out_channels=out_channels,
                             mid_act=mid_act,
                             out_act=out_act,
                             resnet=resnet)

        else:
            backbone = opt['model']['backbone']
            if backbone == 'vanilla':
                self.model = SSN(in_channels=in_channels,
                                 out_channels=out_channels,
                                 mid_act=mid_act,
                                 out_act=out_act,
                                 resnet=resnet)
            elif backbone == 'SSN_v1':
                self.model = SSN_v1(in_channels=in_channels,
                                    out_channels=out_channels,
                                    mid_act=mid_act,
                                    out_act=out_act,
                                    resnet=resnet)
            else:
                raise NotImplementedError('{} has not implemented yet'.format(backbone))


        self.optimizer = get_optimizer(opt, self.model)
        self.visualization = {}

        self.norm_loss = norm_loss()

        # inference related
        BINs    = 100
        MAX_RAD = 20
        self.size_interval     = MAX_RAD / BINs
        self.soft_distribution = [[np.exp(-0.2 * (i - j) ** 2) for i in np.arange(BINs)] for j in np.arange(BINs)]

    def setup_input(self, x):
        return x


    def forward(self, x):
        x, softness = x
        return self.model(x, softness)


    def compute_loss(self, y, pred):
        b = y.shape[0]

        total_loss = self.norm_loss.loss(y, pred)

        if self.focal:
            total_loss = torch.pow(total_loss, 3)

        return total_loss


    def supervise(self, input_x, y, is_training:bool)->float:
        optimizer = self.optimizer
        model = self.model

        x, softness = input_x['x'], input_x['softness']

        optimizer.zero_grad()
        pred = model(x, softness)
        loss = self.compute_loss(y, pred)

        if is_training:
            loss.backward()
            optimizer.step()

        xc = x.shape[1]
        for i in range(xc):
            self.visualization['x{}'.format(i)] = x[:, i:i+1].detach()

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
            ret_vis[k] = np.clip(utils.make_grid(plot_v.cpu(), nrow=nrows).numpy().transpose(1,2,0), 0.0, 1.0)
            ret_vis[k] = self.plasma(ret_vis[k])

        return ret_vis


    def get_logs(self):
        pass


    def inference(self, x):
        x, l, device = x['x'], x['l'], x['device']

        x = torch.from_numpy(x.transpose((2,0,1))).unsqueeze(dim=0).to(device)
        l = torch.from_numpy(np.array(self.soft_distribution[int(l/self.size_interval)]).astype(np.float32)).unsqueeze(dim=0).to(device)

        pred = self.forward((x, l))
        pred = pred[0].detach().cpu().numpy().transpose((1,2,0))
        return pred


    def batch_inference(self, x):
        x, l = x['x'], x['softness']
        pred = self.forward((x, l))
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
