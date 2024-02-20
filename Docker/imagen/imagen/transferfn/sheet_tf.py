"""
A family of transfer functions that are aware of sheet coordinate
systems.

The transfer functions in this file are allowed to make use of Imagen
patterns and are to be supplied with an appropriate
SheetCoordinateSystem object via the initialize method.
"""

import numpy as np

import param
import copy
from imagen import PatternGenerator, Gaussian
from imagen.transferfn import TransferFn


class Convolve(TransferFn):
    """
    Transfer function that convolves the array data with the supplied
    kernel pattern.

    The bounds and densities of the supplied kernel pattern do not
    affect the convolution operation. The spatial scale of the
    convolution is determined by the 'size' parameter of the
    kernel. The resulting convolution is applied of a spatial scale
    relative to the overall size of the input, as expressed in
    sheetcoordinates.
    """

    kernel_pattern = param.ClassSelector(PatternGenerator,
                     default=Gaussian(size=0.05,aspect_ratio=1.0), doc="""
      The kernel pattern used in the convolution. The default kernel
      results in an isotropic Gaussian blur.""")

    init_keys = param.List(default=['SCS'], constant=True)

    def __init__(self, **params):
        super(Convolve,self).__init__(**params)


    def initialize(self,  **kwargs):
        super(Convolve, self).initialize(**kwargs)
        scs = kwargs['SCS']
        pattern_copy = copy.deepcopy(self.kernel_pattern)
        pattern_copy.set_matrix_dimensions(self.kernel_pattern.bounds,
                                           scs.xdensity,
                                           scs.ydensity)
        self.kernel = pattern_copy()

    def __call__(self, x):
        if not hasattr(self, 'kernel'):
            raise Exception("Convolve must be initialized before being called.")
        fft1 = np.fft.fft2(x)
        fft2 = np.fft.fft2(self.kernel, s=x.shape)
        convolved_raw = np.fft.ifft2( fft1 * fft2).real

        k_rows, k_cols = self.kernel.shape  # ORIGINAL
        rolled = np.roll(np.roll(convolved_raw, -(k_cols//2), axis=-1), -(k_rows//2), axis=-2)
        convolved = rolled / float(self.kernel.sum())
        x.fill(0.0)
        x+=convolved
