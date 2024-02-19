import numpy as np
import cv2
import random

random.seed(19920208)

def random_kernel():
    ksize = random.randint(1,3)
    kernel = np.ones((ksize, ksize))
    return kernel
    
def random_perturb(img):
    return img
#     perturbed = img.copy()
#     if random.random() < 0.5:
#         perturbed = cv2.erode(perturbed, random_kernel(), iterations = 1)
    
#     if random.random() < 0.5:
#         perturbed = cv2.dilate(perturbed, random_kernel(), iterations = 1)
    
#     cv2.normalize(perturbed, perturbed, 0.0,1.0, cv2.NORM_MINMAX)
#     if len(perturbed.shape) == 2:
#         perturbed = perturbed[:,:,np.newaxis]
#     return perturbed