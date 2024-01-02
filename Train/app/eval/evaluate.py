import sys
sys.path.insert(0, 'app')

import models
import datasets

import pandas as pd
import shutil
import time
import numpy as np
from os.path import join
import os
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import argparse
import logging
from visualize import visualize_eval
from params import parse_configs
import metrics
from loss import norm_loss, grad_loss
from utils import get_fname


def compute_loss(gt, pred):
    """Loss experiment
    """
    _, _, h, w = gt.shape
    norm_fact = 1.0/(h * w)

    l1 = norm_loss(gt, pred, 1)
    l2 = norm_loss(gt, pred, 2)
    grad = grad_loss(gt, pred)
    l1_lambda, l2_lambda, grad_lambda = 0.8, 0.0, 0.2
    return (l1 * l1_lambda + l2 * l2_lambda + grad * grad_lambda) * norm_fact


class evaluate_obj:
    """ Data structure that holds model and dataloader
    """

    def __init__(self, model_configs, weight_fname, ds_hdf5, opath):
        self.model_name = model_configs['model']
        mid_act         = model_configs['mid_act']
        out_act         = model_configs['out_act']

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = get_model(self.model_name, mid_act, out_act).to(self.device)

        # resume model
        checkpoint = torch.load(weight_fname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('{} resume from {}'.format(self.get_model_name(), weight_fname))

        # different model has different dataloaders
        ds_args = {'hdf5': ds_hdf5, 'model': self.model_name}
        self.ds_loader = DataLoader(
            dataloader_test_init(ds_args),
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False)

        self.opath = join(opath, self.get_model_name())
        os.makedirs(self.opath, exist_ok=True)


    def get_model_name(self):
        return self.model_name


    def evaluate(self):
        """  return score = {'L2':, 'ssim':, 'PSNR':}
        """
        def compute_metrics(x, y, pred):
            """ We ignore those occluded regions
            """
            b, c, h, w = x.shape
            mask = x[:, 0:1].clone()

            if mask.min() < -0.5:
                mask[mask > -0.5] = 0.0
                mask[mask < -0.5] = 1.0
            else:
                mask = 1.0 - mask

            y_mask = y * mask
            pred_mask = pred * mask

            loss = compute_loss(y_mask, pred_mask).item()/b
            L2   = metrics.norm_metric(y_mask, pred_mask).item()
            ssim = metrics.ssim_metric(y_mask, pred_mask).item()
            psnr = metrics.PSNR_metric(y_mask, pred_mask).item()

            return {'L2': L2, 'loss': loss, 'ssim': ssim, 'psnr': psnr}

        # return values
        stats_results = []

        self.model.eval()
        torch.set_grad_enabled(False)

        # remove previous evaluation results
        logging.info('Remove previous results: {}'.format(self.opath))

        if os.path.exists(self.opath):
            shutil.rmtree(self.opath)

        os.makedirs(self.opath, exist_ok=True)

        loss, L2, ssim, psnr = 0, 0, 0, 0
        device = self.device

        for i, data in enumerate(
                tqdm(self.ds_loader, total=len(self.ds_loader),
                     desc='Eval {}'.format(self.get_model_name()))):

            (x, softness) = data['x']
            y             = data['y']
            light_list    = data['light']
            shading       = data['shading']

            x, softness, y = x.to(device), softness.to(device), y.to(device)

            pred = self.model(x, softness)
            pred = torch.clip(pred, 0.0, 1.0)

            if torch.isnan(pred).any():
                import pdb; pdb.set_trace()
                logging.error('{} Prediction find nan'.format(data['path']))
                exit()

            # save results
            metric_result = compute_metrics(x, y, pred)

            L2   += metric_result['L2']
            loss += metric_result['loss']
            ssim += metric_result['ssim']
            psnr += metric_result['psnr']

            y_np  = pred[0].detach().cpu().numpy().transpose((1, 2, 0))
            opath = join(self.opath, '{:05d}.npz'.format(i))
            np.savez_compressed(opath,
                                pred=y_np,
                                loss=metric_result['loss'])

            # save stats
            stats_results.append([
                opath,
                light_list[0].tolist(),
                metric_result['L2'],
                metric_result['loss'],
                metric_result['ssim'],
                metric_result['psnr']])


        N = len(self.ds_loader)
        L2, ssim, psnr = L2/(N), ssim/N, psnr/N
        loss = loss/N

        score = {'loss': loss, 'L2': L2, 'SSIM': ssim, 'PSNR': psnr}
        return score, stats_results


def evaluate(eval_targets):
    ret = {}

    for i, eval_target in enumerate(eval_targets):
        name = eval_target.get_model_name()
        logging.info('Begin evaluate for {}'.format(name))

        ret[name] = eval_target.evaluate()

    return ret


def parsing_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='Visualization output folder')
    params = parser.parse_args()
    configs = parse_configs(params.config)
    logging.info('Evaluation params: {}.\n{}'.format(params, configs))
    return configs


def parse_targets(configs):
    # input check
    ds_hdf5 = configs['Dataset']['DS_hdf5']
    opath   = configs['Output']['tmp']
    models  = configs['Models']

    evaluate_lists = [evaluate_obj({'model': k,
                                    'mid_act': v['mid_act'],
                                    'out_act': v['out_act']},
                                   v['weight'],
                                   ds_hdf5,
                                   opath=opath)
                      for k, v in models.items()]

    return evaluate_lists


def evaluate_methods(configs):
    s = time.time()

    # stats
    stats_data = []

    # parameter initializ
    eval_targets = parse_targets(configs)

    # run the evaluation
    results = evaluate(eval_targets)

    logging.info('--------------Summary---------------')
    for k, v in results.items():
        logging.info('Model: {}\n,{}'.format(k, v[0]))
        stats_data += v[1]
    logging.info('------------------------------------')

    # save stats
    csv_output = os.path.abspath(configs['Output']['stats'])
    dirname    = os.path.dirname(csv_output)
    os.makedirs(dirname, exist_ok=True)

    df = pd.DataFrame(data=stats_data, columns=['path', 'light', 'L2', 'Loss', 'ssim', 'PSNR'])
    df.to_csv(csv_output, index=False)

    logging.info('({}s) Evaluation finished.'.format(time.time() - s))


def render_visualize_videos(configs):
    # IO initialize
    tmp_folder = configs['Output']['tmp']
    video_out  = configs['Output']['video']

    os.makedirs(tmp_folder, exist_ok=True)

    # dump the inputs first
    ds_hdf5    = configs['Dataset']['DS_hdf5']
    ds_args = {'hdf5': ds_hdf5, 'model': 'GSSN_SB_INV'}
    dataset = GSSN_Testing_Dataset(ds_args)

    xy_folder = join(tmp_folder, 'data')

    # import pdb; pdb.set_trace()

    if not os.path.exists(xy_folder):
        for i in tqdm(range(len(dataset))):
            data        = dataset[i]
            x, softness = data['x']
            y           = data['y']
            shading     = data['shading']

            ofile = join(xy_folder, '{:05d}.npz'.format(i))
            np.savez_compressed(ofile,
                                x = x.detach().cpu().numpy().transpose((1,2,0)),
                                y = y.detach().cpu().numpy().transpose((1,2,0)),
                                shading = shading.detach().cpu().numpy().transpose((2,0,1)))


    # make visualization
    visualize_eval(configs)
    logging.info('Visualization rendering finished, refer to {}'.format(video_out))


if __name__ == '__main__':
    log_file = 'logs/{}.log'.format(os.path.splitext(__file__)[0])
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s - %(filename)s %(funcName)s %(asctime)s ",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    configs = parsing_params()

    # evaluate the results
    evaluate_methods(configs)

    # render visualization video
    logging.info('Evaluating finished. Render the video.')
    render_visualize_videos(configs)
