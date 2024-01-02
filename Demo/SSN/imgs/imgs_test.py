import matplotlib.pyplot as plt
import numpy as np


rgb = 'woman.png'
mask = 'woman_mask.png'
ofile = 'test1.png'

rgb = plt.imread(rgb)
mask = plt.imread(mask)

output = np.concatenate([rgb[..., :3], mask[..., :1]], axis=2)
plt.imsave(ofile, output)

