import torch
from torch import nn
import logging

from pathlib import Path
import gradio as gr
import numpy as np
import cv2

import model_utils
from models.SSN import SSN

config_file = 'configs/SSN.yaml'
weight      = 'weights/0000001000.pt'
device      = torch.device('cuda:0')
device      = torch.device('cpu')
model       = model_utils.load_model(config_file, weight, SSN, device)

DEFAULT_INTENSITY = 0.9
DEFAULT_GAMMA = 2.0 

logging.info('Model loading succeed')

cur_rgba = None
cur_shadow = None
cur_intensity = DEFAULT_INTENSITY 
cur_gamma = DEFAULT_GAMMA 

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
    padding = 40
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
    
    padding_h = (256 - new_h) // 2
    padding_w = (256 - new_w) // 2

    padding_output[padding_h:padding_h+new_h, padding_w:padding_w+new_w, :] = padded_rgba
    padding_output = np.clip(padding_output, 0.0, 1.0)

    return padding_output

def shadow_composite(rgba, shadow, intensity, gamma):
    rgb = rgba[..., :3]
    mask = rgba[..., 3:]

    if len(shadow.shape) == 2:
        shadow = shadow[..., None]

    new_shadow = 1.0 - shadow ** gamma * intensity
    ret = rgb * mask + (1.0 - mask) * new_shadow
    return ret, new_shadow[..., 0]


def render_btn_fn(mask, ibl):
    global cur_rgba, cur_shadow, cur_gamma, cur_intensity

    print("Button clicked!")

    mask = mask / 255.0
    ibl = ibl/ 255.0

    mask = np.clip(mask, 0.0, 1.0)

    # smoothing ibl
    ibl = cv2.GaussianBlur(ibl, (11, 11), 0)

    # padding mask
    mask = padding_mask(mask)

    cur_rgba = np.copy(mask)


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
    cur_shadow = np.copy(shadow)

    ret, shadow = shadow_composite(cur_rgba, shadow, cur_intensity, cur_gamma)

    # print('IBL range: {}/{} Shadow range: {} {}'.format(ibl.min(), ibl.max(), shadow.min(), shadow.max()))
    return ret, shadow


def intensity_change(x):
    global cur_rgba, cur_shadow, cur_gamma, cur_intensity

    cur_intensity = x
    ret, shadow = shadow_composite(cur_rgba, cur_shadow, cur_intensity, cur_gamma)    
    return ret, shadow


def gamma_change(x):
    global cur_rgba, cur_shadow, cur_gamma, cur_intensity

    cur_gamma = x
    ret, shadow = shadow_composite(cur_rgba, cur_shadow, cur_intensity, cur_gamma)    
    return ret, shadow

def update_input(mask):
    return mask


ibl_h = 128
ibl_w = ibl_h * 2

with gr.Blocks() as demo:
    with gr.Row():
        mask_input = gr.Image(shape=None, width=256, height=256,image_mode="RGBA", label="RGBA")
        ibl_input = gr.Sketchpad(shape=(ibl_w, ibl_h), image_mode="L", label="IBL", tool='sketch', invert_colors=True)
        output = gr.Image(shape=(256, 256), height=256, width=256, image_mode="RGB", label="Output")
        shadow_output = gr.Image(shape=(256, 256), height=256, width=256, image_mode="L", label="Shadow Layer")

    with gr.Row():
        intensity_slider = gr.Slider(0.0, 1.0, value=DEFAULT_INTENSITY, step=0.1, label="Intensity", info="Choose between 0.0 and 1.0") 
        gamma_slider = gr.Slider(1.0, 4.0, value=DEFAULT_GAMMA, step=0.1, label="Gamma", info="Gamma correction for shadow") 
        render_btn = gr.Button(label="Render")

    with gr.Row():
        gr.Examples(
            examples=[['imgs/woman.png'],['imgs/man.png'], ['imgs/plant1.png'], ['imgs/human2.png'], ['imgs/cloud.png']],
            fn=update_input, 
            inputs=[mask_input],
            outputs=mask_input
        )

    render_btn.click(render_btn_fn, inputs=[mask_input, ibl_input], outputs=[output, shadow_output])
    intensity_slider.release(intensity_change, inputs=[intensity_slider], outputs=[output, shadow_output])
    gamma_slider.release(gamma_change, inputs=[gamma_slider], outputs=[output, shadow_output])

    logging.info('Finished')


demo.launch()
