import random
import time
import numbergen as ng
import imagen as ig
import numpy as np
import cv2
from param.parameterized import get_logger
import logging

get_logger().setLevel(logging.ERROR)

class random_pattern():
    def __init__(self, maximum_blob=50):
        #         self.generator_list = []

        #         start = time.time()
        #         for i in range(maximum_blob):
        #             self.generator_list.append(ig.Gaussian(size=))
        #         print('random pattern init time: {}s'.format(time.time()-start))

        pass

    def y_transform(self, y):
        # y = []
        pass

    def get_pattern(self, w, h, x_density=512, y_density=128, num=50, scale=3.0, size=0.1, energy=3500,
                    mitsuba=False, seed=None, dataset=False):
        if seed is None:
            seed = random.randint(0, 19920208)
        else:
            seed = seed + int(time.time())

        if num == 0:
            ibl = np.zeros((y_density, x_density))
            orientation = np.pi * ng.UniformRandom(seed=seed + 3)()
        else:
            y_fact = y_density / 256
            num = 1
            size = size * ng.UniformRandom(seed=seed + 4)()
            orientation = np.pi * ng.UniformRandom(seed=seed + 3)()
            gs = ig.Composite(operator=np.add,
                              generators=[ig.Gaussian(
                                  size=size,
                                  scale=1.0,
                                  x=ng.UniformRandom(seed=seed + i + 1) - 0.5,
                                  y=((1.0 - ng.UniformRandom(seed=seed + i + 2) * y_fact) - 0.5),
                                  aspect_ratio=0.7,
                                  orientation=orientation,
                              ) for i in range(num)],
                              position=(0, 0),
                              xdensity=512)

            # gs = ig.Composite(operator=np.add,
            #                   generators=[ig.Gaussian(
            #                       size=size * ng.UniformRandom(seed=seed + i + 4),
            #                       scale=scale * (ng.UniformRandom(seed=seed + i + 5) + 1e-3),
            #                       x=int(ind / h),
            #                       y=ind % h,
            #                       aspect_ratio=0.7,
            #                       orientation=np.pi * ng.UniformRandom(seed=seed + i + 3),
            #                   ) for i in range(num)],
            #                   position=(0, 0),
            #                   xdensity=512)
            ibl = gs()[:y_density, :]

        # prepare to fix energy inconsistent
        if dataset:
            ibl = self.to_dataset(ibl, w, h)

        if mitsuba:
            return ibl, size, orientation
        else:
            return ibl, size, orientation

    def to_mts_ibl(self, ibl):
        """ Input: 256 x 512 pattern generated ibl 
            Output: the ibl in mitsuba ibl
        """
        return np.repeat(ibl[:, :, np.newaxis], 3, axis=2)

    def normalize(self, ibl, energy=30.0):
        total_energy = np.sum(ibl)
        if total_energy < 1e-3:
            print('small energy: ', total_energy)
            h, w = ibl.shape
            return np.zeros((h, w))

        return ibl * energy / total_energy

    def to_dataset(self, ibl, w, h):
        return self.normalize(cv2.flip(cv2.resize(ibl, (w, h)), 0), 30)
