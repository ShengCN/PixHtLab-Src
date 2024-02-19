from pathlib import Path
import torch
import gradio as gr
from torch import nn
import numpy as np


def render_btn_fn(mask, background, buffers, pitch, roll, softness):
    print('Pitch and roll: {}, {}'.format(pitch, roll))
    print('Mask, background, bufferss: {}, {}, {}'.format(mask.shape, background.shape, buffers.shape))
    pass


with gr.Blocks() as demo:
    with gr.Row():
        mask_input = gr.Image(shape=(256, 256), image_mode="L", label="Mask")
        bg_input   = gr.Image(shape=(256, 256), image_mode="RGB", label="Background")
        buff_input = gr.Image(shape=(256, 256), image_mode="RGB", label="Buffers")

    with gr.Row():
        with gr.Column():
            pitch_input    = gr.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Pitch")
            roll_input     = gr.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Roll")
            softness_input = gr.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Softness")

    render_btn = gr.Button(label="Render")
    output     = gr.Image(shape=(256, 256), image_mode="RGB", label="Output")

    render_btn.click(render_btn_fn, inputs=[mask_input, bg_input, buff_input, pitch_input, roll_input, softness_input], outputs=output)


demo.launch()
