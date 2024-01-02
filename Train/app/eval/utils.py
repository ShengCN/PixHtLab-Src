import os
from os.path import join
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import cv2

def render_video_from_sequence(pattern, opath, framerate=32):
    (
    ffmpeg
    .input(pattern, pattern_type='glob', framerate=framerate)
    .output(opath, movflags='faststart', pix_fmt='yuv420p', vcodec='libx264')
    .overwrite_output()
    .run()
    )

def draw_title(img, title):
    h,w = img.shape[:2]
    pil_img = Image.fromarray((img*255.0).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
    draw.text((0, 10), title, fill='red', font=font)
    return np.array(pil_img)/255.0

def get_fname(path):
    return os.path.splitext(os.path.basename(path))[0]

def draw_plasma(img):
    plasma_cm = plt.get_cmap('plasma')
    return plasma_cm(img)[..., :3]

def resize(img, w, h):
    return cv2.resize(np.squeeze(img), (w, h))
