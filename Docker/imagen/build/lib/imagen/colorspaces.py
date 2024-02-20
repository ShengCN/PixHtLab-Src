"""
Utilities for converting images between various color spaces, such as:

  * RGB (for display on computer monitor red, green, and blue channels)
  * HSV (allowing manipulation of the hue, saturation, and value),
  * LMS (estimates of human long, medium, and short cone responses),
  * LCH (CIE perceptually uniform luminance, chroma (saturation), and hue)
  * LAB (CIE opponent black/white, red/green, blue/yellow axes)
  * XYZ (CIE interchange format)

See http://en.wikipedia.org/wiki/Color_space for more detailed descriptions.

To use these utilities, users should instantiate one of these two classes:

ColorSpace
    Provides a convert(from, to, what) method to perform conversion
    between colorspaces, e.g.  ``convert("rgb", "hsv", X)``, where ``X``
    is assumed to be a numpy.dstack() object with three matching arrays.

FeatureColorConverter
    Declare a set of color spaces to allow external code to work the
    same for any combination of color spaces.  Specifically, declares:

    * image color space (the space in which a dataset of images has
      been stored),
    * working color space (to which the images will be converted),
         e.g. to transform images to a different working dataset, and
    * analysis color space (space in which analyses will be performed)

    These values can be set using::

      color_conversion.image_space="XYZ"    # e.g. RGB, XYZ, LMS
      color_conversion.working_space="RGB" # e.g. RGB, LMS
      color_conversion.analysis_space="HSV" # e.g. HSV, LCH

The other code in this file is primarily implementation for these two
classes, and will rarely need to be used directly.

"""

from math import pi, fmod, floor

import param
import copy
import colorsys
import numpy as np


def _threeDdot_simple(M,a):
    "Return Ma, where M is a 3x3 transformation matrix, for each pixel"

    result = np.empty(a.shape,dtype=a.dtype)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            A = np.array([a[i,j,0],a[i,j,1],a[i,j,2]]).reshape((3,1))
            L = np.dot(M,A)
            result[i,j,0] = L[0]
            result[i,j,1] = L[1]
            result[i,j,2] = L[2]

    return result


def _threeDdot_opt(M,a):
    swapped = a.swapaxes(0,2)
    shape = swapped.shape
    result = np.dot(M,swapped.reshape((3,-1)))
    result.shape = shape
    b = result.swapaxes(2,0)
    # need to do asarray to ensure dtype?
    return np.asarray(b,dtype=a.dtype)

# CB: probably could make a faster version if do aM instead,
# e.g. something like (untested):

#def _threeDdot(M,a):
#    shape = a.shape
#    result = np.dot(a.reshape((-1,3)),M)
#    result.shape = shape
#    return result

threeDdot = _threeDdot_opt


def _abc_to_def_array(ABC,fn):
    shape = ABC[:,:,0].shape
    dtype = ABC.dtype

    DEF = np.zeros(ABC.shape,dtype=dtype)

    for i in range(shape[0]):
        for j in range(shape[1]):
            DEF[i,j,0],DEF[i,j,1],DEF[i,j,2]=fn(ABC[i,j,0],ABC[i,j,1],ABC[i,j,2])

    return DEF


def _rgb_to_hsv_array(RGB):
    """Equivalent to colorsys.rgb_to_hsv, except expects array like :,:,3"""
    return _abc_to_def_array(RGB,colorsys.rgb_to_hsv)


def _hsv_to_rgb_array(HSV):
    """Equivalent to colorsys.hsv_to_rgb, except expects array like :,:,3"""
    return _abc_to_def_array(HSV,colorsys.hsv_to_rgb)


# these aliases can be overriden after loading this file, if
# optimized versions are available
rgb_to_hsv = _rgb_to_hsv_array
hsv_to_rgb = _hsv_to_rgb_array


# Should document where these constants are from.
KAP = 24389/27.0
EPS = 216/24389.0


def xyz_to_lab(XYZ,wp):

    X,Y,Z = np.dsplit(XYZ,3)
    xn,yn,zn = X/wp[0], Y/wp[1], Z/wp[2]

    def f(t):
        t = t.copy() # probably unnecessary!
        t_eps = t>EPS
        t_not_eps = t<=EPS
        t[t_eps] = np.power(t[t_eps], 1.0/3)
        t[t_not_eps] = (KAP*t[t_not_eps]+16.0)/116.
        return t

    fx,fy,fz = f(xn), f(yn), f(zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)

    return np.dstack((L,a,b))


def lab_to_xyz(LAB,wp):

    L,a,b = np.dsplit(LAB,3)
    fy = (L+16)/116.0
    fz = fy - b / 200.0
    fx = a/500.0 + fy

    def finv(y):
        y =copy.copy(y) # CEBALERT: why copy?
        eps3 = EPS**3
        return np.where(y > eps3,
                           np.power(y,3),
                           (116*y-16)/KAP)

    xr, yr, zr = finv(fx), finv(fy), finv(fz)
    return np.dstack((xr*wp[0],yr*wp[1],zr*wp[2]))


def lch_to_lab(LCH):
    L,C,H = np.dsplit(LCH,3)
    return np.dstack( (L,C*np.cos(H),C*np.sin(H)) )


def lab_to_lch(LAB):
    L,A,B = np.dsplit(LAB,3)
    range_ = 2*pi
    x = np.arctan2(B,A)
    return np.dstack( (L, np.hypot(A,B), fmod(x + 2*range_*(1-floor(x/(2*range_))), range_) ) )


def xyz_to_lch(XYZ,whitepoint):
    return lab_to_lch(xyz_to_lab(XYZ,whitepoint))


def lch_to_xyz(LCH,whitepoint):
    return lab_to_xyz(lch_to_lab(LCH),whitepoint)




# Preceding functions started from ceball's colorfns.py file
# the rest started from
# http://projects.scipy.org/scipy/browser/trunk/Lib/sandbox/image/color.py?rev=1698

whitepoints = {'CIE A': ['Normal incandescent', 0.4476, 0.4074],
               'CIE B': ['Direct sunlight', 0.3457, 0.3585],
               'CIE C': ['Average sunlight', 0.3101, 0.3162],
               'CIE E': ['Normalized reference', 1.0/3, 1.0/3],
               'D50':   ['Bright tungsten', 0.3457, 0.3585],
               'D55':   ['Cloudy daylight', 0.3324, 0.3474],
               'D65':   ['Daylight', 0.312713, 0.329016],
               'D75':   ['?', 0.299, 0.3149],
               'D93':   ['low-quality old CRT', 0.2848, 0.2932]
               }


def triwhite(chrwhite):
    x,y = chrwhite
    X = float(x) / y
    Y = 1.0
    Z = (1-x-y)/y
    return X,Y,Z

for key in whitepoints.keys():
    whitepoints[key].append(triwhite(whitepoints[key][1:]))


transforms = {}


# CEBALERT: add reference.  Inverse computed using scipy.linalg.inv.
transforms = {}
transforms['D65'] = sD65 = {}

sD65['rgb_from_xyz'] = np.array([[3.2410,-1.5374,-0.4986],
                                 [-0.9692,1.8760,0.0416],
                                 [0.0556,-0.204,1.0570]])

sD65['xyz_from_rgb'] = np.array([[ 0.41238088,  0.35757284,  0.1804523 ],
                                 [ 0.21261986,  0.71513879,  0.07214994],
                                 [ 0.0193435 ,  0.11921217,  0.95050657]])


# Guth (1980) - SP; L, M, and S normalized to one)
sD65['lms_from_xyz'] = np.array([[0.2435, 0.8524, -0.0516],
                                 [-0.3954, 1.1642, 0.0837],
                                 [0, 0, 0.6225]])

sD65['xyz_from_lms'] = np.array([[1.87616336e+00, -1.37368291e+00, 3.40220544e-01],
                                 [6.37205799e-01, 3.92411765e-01,  5.61517442e-05],
                                 [0.00000000e+00, 0.00000000e+00,  1.60642570e+00]])


### Make LCH like other spaces (0,1)

Lmax = 100.0
Cmax = 360.0 # ? CEBALERT: A,B typically -127 to 128 (wikipedia...), so 360 or so max for C?
Hmax = 2*pi

def xyz_to_lch01(XYZ, whitepoint):
    L,C,H = np.dsplit(xyz_to_lch(XYZ,whitepoint),3)
    L/=Lmax
    C/=Cmax
    H/=Hmax
    return np.dstack((L,C,H))

def lch01_to_xyz(LCH, whitepoint):
    L,C,H = np.dsplit(LCH,3)
    L*=Lmax
    C*=Cmax
    H*=Hmax
    return lch_to_xyz(np.dstack((L,C,H)),whitepoint)



class ColorSpace(param.Parameterized):
    """
    Low-level color conversion. The 'convert' method handles color
    conversion to and from (and through) XYZ, and supports RGB, LCH,
    LMS and HSV.
    """

    whitepoint = param.String(default='D65', doc="""
        Name of whitepoint in lookup table.""")

    transforms = param.Dict(default=transforms,doc="""
        Structure containing the transformation matrices used by this
        Class. See ``transforms`` in this file.""")

    input_limits = param.NumericTuple((0.0,1.0),doc="""
        Upper and lower bounds to verify on input values.""")

    output_limits = param.NumericTuple((0.0,1.0),doc="""
        Upper and lower bounds to enforce on output values.""")

    output_clip = param.ObjectSelector(default='silent',
                                       objects=['silent','warn','error','none'],doc="""
        Action to take when the output value will be clipped.""")

    dtype = param.Parameter(default=np.float32, doc="Datatype to use for result.")


    def convert(self, from_, to, what):
        """
        Convert image or color "what" from "from_" colorpace to "to"
        colorspace.  E.g.: ``convert("rgb", "hsv", X)``, where X is a
        numpy dstack or a color tuple.
        """

        if(from_.lower()==to.lower()):
            return what

        # Check if there exist an optimized function that performs
        # from_to_to conversion
        direct_conversion = '%s_to_%s'%(from_.lower(),to.lower())
        if( hasattr(self, direct_conversion ) ):
            fn = getattr(self, direct_conversion)
            return fn(what)

        from_to_xyz = getattr(self, '%s_to_xyz'%(from_.lower()) )
        xyz_to_to = getattr(self, 'xyz_to_%s'%(to.lower()) )

        return xyz_to_to( from_to_xyz(what) )


    def _triwp(self):
        return whitepoints[self.whitepoint][3]


    def _get_shape(self,a):
        # The shape of the array if it isn't a scalar
        if isinstance(a, np.ndarray) and a.ndim>0:
            return a.shape
        # Tuple support
        try:
            length = len(a)
            return (length,)
        except TypeError:
            return None

    def _put_shape(self,a,shape):
        if shape is None:
            return self.dtype(a)
        else:
            a.shape = shape
            return a

    def _prepare_input(self,a,min_,max_):
        in_shape = self._get_shape(a)
        a = np.array(a,copy=False,ndmin=3,dtype=self.dtype)
        if a.min()<min_ or a.max()>max_:
            raise ValueError('Input out of limits')
        return a, in_shape

    def _clip(self,a,min_limit,max_limit,action='silent'):
        if action=='none':
            return

        if action=='error':
            if a.min()<min_limit or a.max()>max_limit:
                raise ValueError('(%s,%s) outside limits (%s,%s)'
                                 % (a.min(),a.max(),min_limit,max_limit))
        elif action=='warn':
            if a.min()<min_limit or a.max()>max_limit:
                self.warning('(%s,%s) outside limits (%s,%s)' %
                             (a.min(),a.max(),min_limit,max_limit))
        a.clip(min_limit,max_limit,out=a)


    def _threeDdot(self,M,a):
        # b = Ma
        a, in_shape = self._prepare_input(a,*self.input_limits)
        b = threeDdot(M,a)
        self._clip(b,*self.output_limits,action=self.output_clip)
        self._put_shape(b,in_shape)
        return b

    def _ABC_to_DEF_by_fn(self,ABC,fn,*fnargs):
        ABC, in_shape = self._prepare_input(ABC,*self.input_limits)
        DEF = fn(ABC,*fnargs)
        self._clip(DEF,*self.output_limits,action=self.output_clip)
        self._put_shape(DEF, in_shape)
        return DEF

    ##  TO XYZ:     RGB, LCH, LMS, HSV(passing through RGB)
    def rgb_to_xyz(self,RGB):
        return self._threeDdot(
            self.transforms[self.whitepoint]['xyz_from_rgb'], RGB)


    def lch_to_xyz(self,LCH):
        return self._ABC_to_DEF_by_fn(LCH,lch01_to_xyz,self._triwp())


    def lms_to_xyz(self,LMS):
        return self._threeDdot(
            self.transforms[self.whitepoint]['xyz_from_lms'], LMS)


    def hsv_to_xyz(self,HSV):
        return self.rgb_to_xyz(self.hsv_to_rgb(HSV))

    ##  XYZ TO:RGB, LCH, LMS, HSV(passing through RGB)

    def xyz_to_rgb(self,XYZ):
        return self._threeDdot(
            self.transforms[self.whitepoint]['rgb_from_xyz'], XYZ)


    def xyz_to_lch(self, XYZ):
        return self._ABC_to_DEF_by_fn(XYZ,xyz_to_lch01,self._triwp())


    def xyz_to_lms(self,XYZ):
        return self._threeDdot(
            self.transforms[self.whitepoint]['lms_from_xyz'], XYZ)


    def xyz_to_hsv(self, XYZ):
        return self.rgb_to_hsv( self.xyz_to_rgb(XYZ) )

    # Optimized
    @staticmethod
    def _gamma_rgb(RGB):
        return (12.92*RGB*(RGB<=0.0031308)
                + ((1+0.055)*RGB**(1/2.4) - 0.055) * (RGB>0.0031308))

    @staticmethod
    def _ungamma_rgb(RGB):
        return (RGB/12.92*(RGB<=0.04045)
                + (((RGB+0.055)/1.055)**2.4) * (RGB>0.04045))

    def rgb_to_hsv(self,RGB):
        "linear rgb to hsv"
        gammaRGB = self._gamma_rgb(RGB)
        return self._ABC_to_DEF_by_fn(gammaRGB,rgb_to_hsv)

    def hsv_to_rgb(self,HSV):
        "hsv to linear rgb"
        gammaRGB = self._ABC_to_DEF_by_fn(HSV,hsv_to_rgb)
        return self._ungamma_rgb(gammaRGB)

    def hsv_to_gammargb(self,HSV):
        "hsv is already specifying gamma corrected rgb"
        return self._ABC_to_DEF_by_fn(HSV,hsv_to_rgb)

    def lch_to_gammargb(self,LCH):
        return self._gamma_rgb(self.lch_to_rgb(LCH))

    def lms_to_lch(self,LCH):
        lch_to_xyz



def _swaplch(LCH):
    "Reverse the order of an LCH numpy dstack or tuple for analysis."
    try: # Numpy array
        L,C,H = np.dsplit(LCH,3)
        return np.dstack((H,C,L))
    except: # Tuple
        L,C,H = LCH
        return H,C,L



class ColorConverter(param.Parameterized):
    """
    High-level color conversion class designed to support color space
    transformations along a pipeline common in color vision modelling:
    image (dataset colorspace) -> working (working colorspace) ->
    [higher stages] -> analysis
    """

    # CEBALERT: should be ClassSelector

    # SPG: it shouldn't be necessary to support selection of the
    # ColorSpace, as the new object supports any color conversion.
    colorspace = param.Parameter(default=ColorSpace(),doc="""
        Object to use for converting between color spaces.""")

    image_space = param.ObjectSelector(default='XYZ', objects=['XYZ', 'LMS', 'RGB'], doc="""
        Color space in which images are encoded.""") # CEBALERT: possibly add sRGB?

    working_space = param.ObjectSelector(default='RGB', objects=['RGB','LMS'], doc="""
        Color space to which images will be transformed to provide
        working space to later stages of processing.""")

    analysis_space = param.ObjectSelector(default='HSV',
                                          objects=['HSV','LCH'], doc="""
        Color space in which analysis is performed.""")

    swap_polar_HSVorder = {
        'HSV': lambda HSV: HSV,
        'LCH': _swaplch }


    def image2working(self,i):
        """Transform images i provided into the specified working
        color space."""
        return self.colorspace.convert(self.image_space,
                                       self.working_space, i)

    def working2analysis(self,r):
        "Transform working space inputs to the analysis color space."
        a = self.colorspace.convert(self.working_space, self.analysis_space, r)
        return self.swap_polar_HSVorder[self.analysis_space](a)

    def analysis2working(self,a):
        "Convert back from the analysis color space to the working space."
        a = self.swap_polar_HSVorder[self.analysis_space](a)
        return self.colorspace.convert(self.analysis_space, self.working_space, a)

    def analysis2display(self,a):
        """
        Utility conversion function that transforms data from the
        analysis color space to the display space (currently hard-set
        to RGB) for visualization.
        """
        a = self.swap_polar_HSVorder[self.analysis_space](a)
        return self.colorspace.convert(self.analysis_space.lower(), 'gammargb', a)


    def jitter_hue(self,a,amount):
        "Rotate the hue component of a by the given amount."
        a[:,:,0] += amount
        a[:,:,0] %= 1.0

    def multiply_sat(self,a,factor):
        "Scale the saturation of a by the given amount."
        a[:,:,1] *= factor


# Provide a shared color_conversion object
color_conversion = ColorConverter()


__all__ = ["ColorSpace","ColorConverter"]
