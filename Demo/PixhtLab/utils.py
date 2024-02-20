import os
import sys
from os.path import join

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget, QAction, QFileDialog,QLabel, QPushButton, QSlider, QGridLayout, QGroupBox, QListWidget
from PyQt5.QtGui import QIcon, QPixmap, QImage

import cv2
import numpy as np
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt
import time

def lerp(a, b, t):
	return (1.0-t) * a +  t * b

def resize(img, max_size):
	old_shape = len(img.shape)
	h,w = img.shape[:2]
	if h > w:
		newh, neww = max_size, int(max_size * w/h)
	else:
		newh, neww = int(max_size * h / w), max_size
	ret = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
	if old_shape != len(ret.shape):
		return ret[..., np.newaxis]
	return ret

def set_qt_img(img, label):
	pixmap = QPixmap(img)
	label.setPixmap(pixmap)
	label.adjustSize()

def to_qt_img(np_img):
	if np_img.dtype != np.uint8:
		np_img = np.clip(np_img, 0.0, 1.0)
		np_img = np_img * 255.0
		np_img = np_img.astype(np.uint8)

	if len(np_img.shape) == 2:
		np_img = np_img[..., np.newaxis].repeat(3, axis=2)

	h, w, c = np_img.shape
	# bytesPerLine = 3 * w
	return QImage(np_img.data, w, h, 3 * w, QImage.Format_RGB888)

def update_img_widget(widget, img):
	set_qt_img(to_qt_img(img), widget)

def overlap_replace(a, b, start_pos):
	""" overlapping a and b 
	"""
	ha, wa = a.shape[:2]
	hb, wb = b.shape[:2]
	a_ = a.copy()

	sh, shh, sw, sww = start_pos[0], start_pos[0] + hb, start_pos[1], start_pos[1] + wb
	clipped_h, clipped_hh, clipped_w, clipped_ww = np.clip(sh,0,ha), np.clip(shh, 0, ha), np.clip(sw, 0, wa), np.clip(sww, 0, wa)
	h,w = clipped_hh - clipped_h, clipped_ww - clipped_w
	a_[clipped_h:clipped_hh, clipped_w:clipped_ww] = b[clipped_h-sh:clipped_h-sh+h, clipped_w-sw:clipped_w-sw+w]
	return a_

def composite(a, b, bmask):
	return (1.0-bmask) * a  + bmask * b

def overlap_comp(a, b, bmask, start_pos):
	""" overlap and composite a and b with bmask 
	"""
	acopy = a.copy()
	ha, wa = a.shape[:2]
	hb, wb = b.shape[:2]

	sh, shh, sw, sww = start_pos[0], start_pos[0] + hb, start_pos[1], start_pos[1] + wb
	clipped_h, clipped_hh, clipped_w, clipped_ww = np.clip(sh,0,ha), np.clip(shh, 0, ha), np.clip(sw, 0, wa), np.clip(sww, 0, wa)
	h,w = clipped_hh - clipped_h, clipped_ww - clipped_w
	acopy[clipped_h:clipped_hh, clipped_w:clipped_ww] = composite(a[clipped_h:clipped_hh, clipped_w:clipped_ww], b[clipped_h-sh:clipped_h-sh+h, clipped_w-sw:clipped_w-sw+w], bmask[clipped_h-sh:clipped_h-sh+h, clipped_w-sw:clipped_w-sw+w]) 
	return acopy

def line_height(a, b, x):
    # line equation:
    # f = a + (b-a) * t 
    # t = (f-a)/(b-a)
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    t = (x-ax)/(bx-ax)
    y = ay + (by-ay) * t
    return y

def line_height_map(p0, p1, w, h):
#     start, end = np.array([384, 491]), np.array([2176, 938])
    start, end = p0, p1
    img = np.zeros((h,w,3))
    line_vec = end - start
    x,y = np.arange(0, w), np.arange(0,h)
    xx, yy = np.meshgrid(x, y)
    height = line_height(start, end, xx)
    yy = height - yy
    yy = yy / h
    yy = np.clip(yy, 0.0, 1.0)
    return yy

def heightmap_resize(hmap, ori_mask, newsize):
	""" Dilate original heightmap  
	"""
	oh, ow = hmap.shape[:2]
	black_size = max(ow, oh)

	# Filling the hmap by dilation 		
	kernel = np.ones((black_size,black_size), np.uint8)
	uintmask = (ori_mask*255.0).astype(np.uint8)
	dilated_mask = cv2.dilate(uintmask, kernel, 1)
	if len(dilated_mask.shape) != len(uintmask.shape):
		dilated_mask = dilated_mask[..., np.newaxis]
	inpainting_mask = dilated_mask - uintmask

	inpainted_hmap = cv2.inpaint((hmap * 255.0).astype(np.uint8), inpainting_mask[...,0], 3, cv2.INPAINT_TELEA)
	resized_hmap = resize(inpainted_hmap, newsize)/255.0
	if len(resized_hmap.shape) == 2: 
		resized_hmap = resized_hmap[..., np.newaxis]	
	return resized_hmap 

def heightmap_resizehw(hmap, ori_mask, neww, newh):
	""" Dilate original heightmap  
	"""
	oh, ow = hmap.shape[:2]
	black_size = max(ow, oh)

	# Filling the hmap by dilation 		
	kernel = np.ones((black_size,black_size), np.uint8)
	uintmask = (ori_mask*255.0).astype(np.uint8)
	dilated_mask = cv2.dilate(uintmask, kernel, 1)
	if len(dilated_mask.shape) != len(uintmask.shape):
		dilated_mask = dilated_mask[..., np.newaxis]
	inpainting_mask = dilated_mask - uintmask
	inpainted_hmap = cv2.inpaint((hmap * 255.0).astype(np.uint8), inpainting_mask[...,0], 3, cv2.INPAINT_TELEA)

	resized_hmap = cv2.resize(inpainted_hmap, (neww, newh))/255.0
	if len(resized_hmap.shape) == 2: 
		resized_hmap = resized_hmap[..., np.newaxis]	
	return resized_hmap 


def read_img(fname, fmt='RGB'):
	return np.array(Image.open(fname).convert(fmt))/255.0

def save_img(fname, img):
	plt.imsave(fname, np.clip(img, 0.0, 1.0))
	print('{} file saved'.format(fname))

def draw_line(img, p0, p1, color='red'):
	pil_img = Image.fromarray((img*255.0).astype(np.uint8))
	img_draw = ImageDraw.Draw(pil_img) 
	img_draw.line((p0, p1), fill=color, width=2)
	return np.array(pil_img)/255.0

def draw_point(img, p, size=5, color='red'):
	pil_img = Image.fromarray((img*255.0).astype(np.uint8))
	img_draw = ImageDraw.Draw(pil_img)
	img_draw.ellipse((p[0], p[1], p[0] + size, p[1] + size), fill=color)
	return np.array(pil_img)/255.0


def compute_wall_hmap(wall_line, rechmap):
	h,w = rechmap.shape[:2]
	newrechmap = line_height_map(wall_line[0], wall_line[1], w, h)	
	newrechmap = np.repeat(newrechmap[..., np.newaxis], 3, axis=2)
	return newrechmap * h

def visualize_hmap(rechmap):
	h,w = rechmap.shape[:2]
	rgb1, rgb2 = np.array([[2, 245, 229]])/255.0, np.array([[70, 212, 202]])/255.0

	rechmap = rechmap/h
	wall_mask = rechmap.copy()
	wall_mask[wall_mask>0] = 1.0
	vis_layer = (1.0-wall_mask) * rgb1 + wall_mask * rgb2
	return vis_layer
