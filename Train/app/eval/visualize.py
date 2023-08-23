import os
from os.path import join
import numpy as np
import logging
import cv2
from torch.utils.data import DataLoader
from GSSN_Dataset import GSSN_Testing_Dataset

import sys
sys.path.insert(0, 'app')
sys.path.insert(0, 'app/Evaluate')

from utils import render_video_from_sequence, draw_title,get_fname, draw_plasma, resize
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from glob import glob

def vis_shadow(x, shadow, shading):
    mask = x[..., 0:1].copy()

    if mask.min() < 0.0:
        mask[mask>-0.9] = 1.0
        mask[mask<0.0] = 0.0

    mask = np.repeat(mask, 3, axis=2)

    shadow_comp = mask * shading + (1.0-mask) * shading * shadow
    return shadow_comp


def set_image(table_img, img, row, col, h, w):
    table_img[row * h:(row+1)* h, col * w: (col+1) * w] = img


def get_plot(diff_dict):
    # plot current averaged diff
    names, values = list(diff_dict.keys()), list(diff_dict.values())

    fig = plt.figure()

    plt.bar(names, values)
    fig.canvas.draw()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(nrows, ncols, 3)/255.0

    plt.close()
    return image


def draw_comps(vis_img, x, y, shading, pred_dict):
    # visualize ground truth
    h, w = x.shape[:2]
    shadow = np.repeat(1.0-y, 3, axis=2)
    shadow_comp = vis_shadow(x, shadow, shading)
    set_image(vis_img, draw_title(shadow_comp, 'GT'), 1, 0, h, w)

    # visualize predictions
    for i, k in enumerate(pred_dict.keys()):
        pred = pred_dict[k]
        diff = np.abs(y-pred)

        norm_l1 = diff.sum()

        if np.isnan(norm_l1):
            logging.error('Find nan. {}'.format(k))
            exit()

        shadow      = np.repeat(1.0-pred, 3, axis=2)
        shadow_comp = vis_shadow(x, shadow, shading)

        set_image(vis_img, draw_title(shadow_comp, '{}_{:.2f}'.format(k, norm_l1)), 1, i+1, h, w)

    # set_image(vis_img, cv2.resize(get_plot(diff_dict), (w, h)), 1, (len(pred_dict.keys()) + 1), h, w)


def draw_diffs(vis_img, x, y, shading, pred_dict):
    def vis_color_map(img):
        vis_cm = plt.get_cmap('RdBu')
        vis_range = (np.squeeze(img) + 1.0) * 0.5
        return vis_cm(vis_range)[..., :3]

    h, w = y.shape[:2]

    mask = 1.0 - x[..., 0:1]

    reference_method = 'SSG'
    ref_diff = np.abs(pred_dict[reference_method]-y)

    for i, k in enumerate(pred_dict.keys()):
        if k == reference_method:
            continue

        diff = np.abs(y-pred_dict[k])
        norm_l1 = diff.sum()
        cur_variation = ref_diff - diff # -1 ~ 1
        scale_variation = np.sign(cur_variation) * np.power(np.abs(cur_variation), 0.4)
        vis_variation = vis_color_map(scale_variation) * mask

        vis_comp = vis_variation * mask + (1.0-mask) * shading

        set_image(vis_img,
                  draw_title(vis_comp, '{}_{:.2f}'.format(k, norm_l1)),
                  2, i + 1, h, w)

def draw_inputs(vis_img, x):
    h, w, c = x.shape
    # fill in x
    for i in range(c):
        # visualize x via plasma
        colored_image = draw_plasma(x[..., i])
        set_image(vis_img, colored_image[..., :3], 0, i, h, w)


def assemble_visualize(x, y, pred_dict, shading):
    """ Given x,y, a list of prediction results, and statistics of the difference
        Return the visualized result
    """
    comp_n = len(pred_dict.keys())
    h, w, c = x.shape
    cols = max(comp_n + 1, c)
    plasma_cm = plt.get_cmap('plasma')

    vis_img = np.zeros((3 * h, cols * w, 3))

    # first row
    draw_inputs(vis_img, x)

    # second row
    # draw_comps(vis_img, x, y, shading, pred_dict, diff_dict)
    draw_comps(vis_img, x, y, shading, pred_dict)

    # third row
    draw_diffs(vis_img, x, y, shading, pred_dict)

    return vis_img


def read_predictions(tmp_output, fname, model_list):
    pred_dict = {}
    diff_dict = {}
    for model_name in model_list:
        data = np.load(join(tmp_output, model_name, fname + '.npz'))

        pred_dict[model_name] = data['pred']
        diff_dict[model_name] = data['loss']

    return pred_dict, diff_dict


def vis_worker(input):
    idx, tmp_output, model_list = input
    vis_tmp_output = join(tmp_output, 'vis')

    xy_folder = join(tmp_output, 'data')
    data      = np.load(join(xy_folder, '{:05d}.npz'.format(idx)))

    cur_x   = data['x']
    cur_y   = data['y']
    shading = data['shading']

    h, w = cur_y.shape[:2]

    cur_shading = shading.transpose((2,0,1))
    cur_shading = resize(cur_shading, w, h)[...,:3]

    fname                    = '{:05d}'.format(idx)
    pred_dict, cur_diff_dict = read_predictions(tmp_output, fname, model_list)
    vis_img                  = assemble_visualize(cur_x, cur_y, pred_dict, cur_shading)
    vis_output               = join(vis_tmp_output, fname + '.png')

    plt.imsave(vis_output, vis_img)

    return cur_diff_dict


def visualize_eval(configs):
    model_list  = configs['Models']
    tmp_output  = configs['Output']['tmp']
    video_out   = configs['Output']['video']

    if len(model_list) == 0:
        err = 'No model privided'
        logging.error(err)
        raise ValueError(err)

    vis_tmp_output = join(tmp_output, 'vis')

    if os.path.exists(vis_tmp_output):
        shutil.rmtree(vis_tmp_output)

    os.makedirs(vis_tmp_output, exist_ok=True)

    # make visualized images
    diff_dict = {model:0 for model in model_list.keys()}  # l1 diff

    file_n = len(glob(join(tmp_output, list(model_list.keys())[0], '*.npz')))

    inputs = [[idx,
               tmp_output,
               list(model_list.keys())] for idx in range(file_n)]

    # if True:
    #     import pdb; pdb.set_trace()
    #     for inp in tqdm(inputs):
    #         cur_diff_dict = vis_worker(inp)

    processer_num = 32
    with multiprocessing.Pool(processer_num) as pool:
        for i, cur_diff_dict in tqdm(enumerate(pool.imap_unordered(vis_worker, inputs), 1), total=len(inputs)):
            for k, v in diff_dict.items():
                diff_dict[k] += cur_diff_dict[k]


    render_video_from_sequence(join(vis_tmp_output, '*.png'), video_out, framerate=8)
