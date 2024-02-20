import time
import torch
import hshadow
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from os.path import join
import numpy as np
from scipy.ndimage import uniform_filter

test_output = 'imgs/output'
os.makedirs(test_output, exist_ok=True)
def test_shadow(rgb, mask, hmap, rechmap, light_pos):
	h,w = rgb.shape[:2]
	bb = torch.tensor([[0, h-1, 0, w-1]]).float().to(device)
	start = time.time()
	shadow = hshadow.forward(rgb, mask, bb, hmap, rechmap, light_pos)[0]
	end = time.time()

	print('Shadow rendering: {}s'.format(end-start))
	res = (1.0-mask) * rgb * shadow + mask * rgb
	return res, shadow

def frenel_reflect(reflect_tensor, reflect_mask, fov, ref_ind):
	# Use schilik approximation https://en.wikipedia.org/wiki/Schlick%27s_approximation
	def deg2rad(deg):
		return deg/180.0 * 3.1415926 

	def img2cos(reflect_img, fov, horizon):
		# Note, this factor needs calibration if we have camera parameters
		b, c, h, w = reflect_img.shape
		focal = 0.5 * h / np.tan(deg2rad(0.5 * fov))
		fadding_map = torch.arange(0, h).unsqueeze(1).expand(h, w).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)
		fadding_map = focal / torch.sqrt((fadding_map-horizon)**2 + focal **2)
		return fadding_map.to(reflect_img)


	ind = (1.0-ref_ind)/(1.0+ref_ind)
	ind = ind ** 2
	h = reflect_tensor.shape[2]
	horizon = h * 0.7 
	cos_map = img2cos(reflect_tensor, fov, horizon)
	# fadding =  1.0 - (ind + (1.0-ind) * torch.pow(1.0-cos_map, 4))
	b, c, h, w = reflect_tensor.shape
	fadding = torch.linspace(3.0,0.0,h)[None, None, ..., None].repeat(b,c,1,w).to(reflect_tensor) ** 4
	fadding = torch.clip(fadding, 0.0, 1.0)
	plt.imsave('test_fadding.png', fadding[0].detach().cpu().numpy().transpose(1,2,0))
	reflect_mask = reflect_mask.repeat(1,3,1,1)
	return fadding * reflect_tensor * reflect_mask + (1.0-reflect_mask * fadding) * torch.ones_like(reflect_tensor) 

def refine_boundary(output, filter=3):
	return output
	h,w,c = output.shape
	for i in range(c):
		output[...,i] = uniform_filter(output[...,i], size=filter)
	return output

def height_fadding(reflect, reflect_height, reflect_mask, fadding_factor):
	def np_sigmoid(a):
		return 1.0/(1.0+np.exp(-a))

	h,w,c = reflect.shape
	reflect_h = reflect_height/h
	reflect_h = reflect_h/reflect_h.max()
	fadding = (1.0-(np_sigmoid(reflect_h * fadding_factor)-0.5)* 2.0) * reflect_mask 
	after_fadding = fadding * reflect + (1.0-fadding) * np.ones_like(reflect)
	return  after_fadding

def to_numpy(tensor):
	return tensor[0].detach().cpu().numpy().transpose(1,2,0)

def test_reflect(rgb, mask, hmap, rechmap, thresholds=1.5, fadding_factor=10.0):
	b, c, h, w = rgb.shape
	# thresholds = torch.tensor([[1.0 + i/b] for i in range(b)]).float().to(device)
	thresholds = torch.tensor([[thresholds]]).float().to(device)
	start = time.time()
	reflect, reflect_height, reflect_mask = hshadow.reflection(rgb, mask, hmap, rechmap, thresholds)
	end = time.time()
	print('Reflection rendering: {}s'.format(end-start))

	reflect, reflect_height, reflect_mask = to_numpy(reflect), to_numpy(reflect_height), to_numpy(reflect_mask)
	# reflect = frenel_reflect(reflect, reflect_height, reflect_mask, 175, 0.9)
	refine_reflect, refine_reflect_height = refine_boundary(reflect), refine_boundary(reflect_height)
	refine_reflect_mask = reflect_mask
	reflect = height_fadding(refine_reflect, refine_reflect_height, refine_reflect_mask, fadding_factor)
	rgb, mask = to_numpy(rgb), to_numpy(mask)
	res = (1.0-mask) * rgb * reflect + mask * rgb
	return res, reflect


def test_glossy_reflect(rgb, mask, hmap, rechmap, sample, glossy, fadding_factor=10.0):
	b, c, h, w = rgb.shape
	# thresholds = torch.tensor([[1.0 + i/b] for i in range(b)]).float().to(device)
	start = time.time()
	reflect = hshadow.glossy_reflection(rgb, mask, hmap, rechmap, sample, glossy)[0]
	end = time.time()
	print('Reflection rendering: {}s'.format(end-start))

	reflect = to_numpy(reflect)
	refine_reflect = refine_boundary(reflect)
	rgb, mask = to_numpy(rgb), to_numpy(mask)
	res = (1.0-mask) * rgb * reflect + mask * rgb
	return res, reflect


device = torch.device("cuda:0")
to_tensor = transforms.ToTensor()
# for i in range(1,5):
i = 2
if True:
	prefix = 'canvas{}'.format(i)
	rgb, mask, hmap = to_tensor(Image.open('imgs/{}_rgb.png'.format(prefix)).convert('RGB')).to(device), to_tensor(Image.open('imgs/{}_mask.png'.format(prefix)).convert('RGB'))[0:1].to(device), to_tensor(Image.open('imgs/{}_height.png'.format(prefix)).convert('RGB'))[0:1].to(device)
	h, w = hmap.shape[1:]
	hmap = hmap * h * 0.45

	rechmap = torch.zeros_like(hmap)
	rgb, mask, hmap, rechmap = rgb.unsqueeze(dim=0), mask.unsqueeze(dim=0), hmap.unsqueeze(dim=0), rechmap.unsqueeze(dim=0)
	lightpos = torch.tensor([[300, -100, 200.0]]).to(device)
	shadow_res, shadow = test_shadow(rgb, mask, hmap, rechmap, lightpos)
	reflect_res, reflect = test_reflect(rgb, mask, hmap, rechmap, thresholds=2.5, fadding_factor=18.0)

	glossy_reflect_res, glossy_reflect = test_glossy_reflect(rgb, mask, hmap, rechmap, sample=10, glossy=0.5, fadding_factor=18.0)

	plt.imsave(join(test_output, prefix + "_shadow_final.png"), shadow_res[0].detach().cpu().numpy().transpose(1,2,0))
	plt.imsave(join(test_output, prefix + "_shadow.png"), shadow[0].detach().cpu().numpy().transpose(1,2,0))
	plt.imsave(join(test_output, prefix + "_reflect_final.png"), reflect_res)
	plt.imsave(join(test_output, prefix + "_reflect.png"), reflect)
	plt.imsave(join(test_output, prefix + "_glossy_reflect_final.png"), glossy_reflect_res)
	plt.imsave(join(test_output, prefix + "_glossy_reflect.png"), glossy_reflect)
