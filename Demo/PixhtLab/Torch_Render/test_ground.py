import time
import torch
import plane_visualize
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from os.path import join
import numpy as np

test_output = 'imgs/output'
os.makedirs(test_output, exist_ok=True)
device = torch.device("cuda:0")

def test_ground():
	fov, horizon = 120, 400 
	camera = torch.tensor([[fov, horizon]])
	planes = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

	camera = camera.repeat(5,1).float().to(device)
	planes = planes.repeat(5,1).float().to(device) 

	ground_vis = plane_visualize.forward(planes, camera, int(512), int(512))[0]
	return ground_vis

t = time.time()
ground_vis = test_ground()
print('{} s'.format(time.time() - t))
batch = ground_vis.shape[0]
for bi in range(batch):
	img = ground_vis[bi].detach().cpu().numpy().transpose(1,2,0)
	img = np.clip(img, 0.0, 1.0)
	plt.imsave(join(test_output, 'ground_{}.png'.format(bi)),img)