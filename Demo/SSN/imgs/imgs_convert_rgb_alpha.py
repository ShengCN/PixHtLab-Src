import matplotlib.pyplot as plt
import numpy as np

rgb_file = 'fg-1-rgb.png'
alpha_file = 'fg-1-alpha.png'
output_file = 'fg-1-rgba.png'

rgb = plt.imread(rgb_file)
alpha = plt.imread(alpha_file)

print(rgb.shape, alpha.shape)

rgba = np.concatenate([rgb[..., :3], alpha[..., 0:1]], axis=2)
plt.imsave(output_file, rgba)
