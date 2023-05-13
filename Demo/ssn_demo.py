from pathlib import Path
import torch
import gradio as gr
from torch import nn
import numpy as np

def render_btn_fn(mask, ibl):
    print("Button clicked!")

    print('mask shape: {}, ibl shape: {}'.format(mask.shape, ibl.shape))

    ret = np.random.randn(256, 256, 3)
    ret = (ret - ret.min()) / (ret.max() - ret.min() + 1e-8)

    return ret


with gr.Blocks() as demo:
    mask_input = gr.Image(shape=(256, 256), image_mode="L", label="Mask")
    ibl_input = gr.Sketchpad(shape=(32, 16), image_mode="L", label="IBL", tool='sketch', invert_colors=False)


    render_btn = gr.Button(label="Render")
    output = gr.Image(shape=(256, 256), image_mode="RGB", label="Output")

    render_btn.click(render_btn_fn, inputs=[mask_input, ibl_input], outputs=output)


demo.launch()
