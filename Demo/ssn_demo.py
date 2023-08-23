import torch
from torch import nn
import logging

from pathlib import Path
import gradio as gr
import numpy as np
import cv2

import model_utils
from models.SSN import SSN

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

config_file = 'configs/SSN.yaml'
weight      = 'weights/SSN/0000001850.pt'
device      = torch.device('cuda:0')
model       = model_utils.load_model(config_file, weight, SSN, device)

logging.info('Model loading succeed')


def resize(img, size):
    h, w = img.shape[:2]

    if h > w:
        newh = size
        neww = int(w / h * size)
    else:
        neww = size
        newh = int(h / w * size)

    resized_img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    if len(img.shape) != len(resized_img.shape):
        resized_img = resized_img[..., none]

    return resized_img


def ibl_normalize(ibl, energy=30.0):
    total_energy = np.sum(ibl)
    if total_energy < 1e-3:
        # print('small energy: ', total_energy)
        h,w = ibl.shape
        return np.zeros((h,w))

    return ibl * energy / total_energy


def padding_mask(rgba_input: np.array):
    """ Padding the mask input so that it fits the training dataset view range

    If the rgba does not have enough padding area, we need to pad the area

    :param rgba_input: H x W x 4 inputs, the first 3 channels are RGB, the last channel is the alpha
    :returns: H x W x 4 padded RGBAD

    """
    padding = 50
    padding_size = 256 - padding * 2

    h, w = rgba_input.shape[:2]
    rgb = rgba_input[:, :, :3]
    alpha = rgba_input[:, :, -1:]

    zeros = np.where(alpha==0)
    hh, ww = zeros[0], zeros[1]
    h_min, h_max = hh.min(), hh.max()
    w_min, w_max = ww.min(), ww.max()

    # if the area already has enough padding
    if h_max - h_min < padding_size and w_max - w_min < padding_size:
        return rgba_input

    padding_output = np.zeros((256, 256, 4))
    padding_output[..., :3] = 1.0

    padded_rgba  = resize(rgba_input, padding_size)
    new_h, new_w = padded_rgba.shape[:2]

    padding_output[padding:padding+new_h, padding:padding+new_w, :] = padded_rgba

    return padding_output



def render_btn_fn(mask, ibl):
    print("Button clicked!")

    mask = mask / 255.0
    ibl = ibl/ 255.0

    # smoothing ibl
    ibl = cv2.GaussianBlur(ibl, (11, 11), 0)

    # padding mask
    mask = padding_mask(mask)

    print('mask shape: {}/{}/{}/{}, ibl shape: {}/{}/{}/{}'.format(mask.shape, mask.dtype, mask.min(), mask.max(),
                                                                   ibl.shape, ibl.dtype, ibl.min(), ibl.max()))

    # ret = np.random.randn(256, 256, 3)
    # ret = (ret - ret.min()) / (ret.max() - ret.min() + 1e-8)

    rgb, mask = mask[..., :3], mask[..., 3]

    ibl = ibl_normalize(cv2.resize(ibl, (32, 16)))

    # ibl = 1.0 - ibl

    x = {
        'mask': mask,
        'ibl': ibl
    }
    shadow = model.inference(x)

    # gamma
    shadow = np.power(shadow, 2.2)
    shadow = shadow * 0.8

    shadow = 1.0 - shadow



    # composite the shadow
    shadow = shadow[..., None]
    mask = mask[..., None]
    ret = rgb * mask + (1.0 - mask) * shadow

    # import pdb; pdb.set_trace()
    # ret = (1.0-mask) * shadow

    print('IBL range: {}/{} Shadow range: {} {}'.format(ibl.min(), ibl.max(), shadow.min(), shadow.max()))

    plt.figure(figsize=(15, 10))
    plt.subplot(1,3,1)
    plt.imshow(mask)
    plt.subplot(1,3,2)
    plt.imshow(ibl)
    plt.subplot(1,3,3)
    plt.imshow(ret)
    plt.savefig('tmp.png')
    plt.close()

    logging.info('Finished')

    return ret


ibl_h = 128
ibl_w = ibl_h * 2

with gr.Blocks() as demo:
    with gr.Row():
        mask_input = gr.Image(shape=(256, 256), image_mode="RGBA", label="Mask")
        ibl_input = gr.Sketchpad(shape=(ibl_w, ibl_h), image_mode="L", label="IBL", tool='sketch', invert_colors=True, bruch_radius=0.1)
        output = gr.Image(shape=(256, 256), image_mode="RGB", label="Output", style="width: 256px; height: 256px;")

    with gr.Row():
        render_btn = gr.Button(label="Render")

    render_btn.click(render_btn_fn, inputs=[mask_input, ibl_input], outputs=output)

    logging.info('Finished')


demo.launch()
