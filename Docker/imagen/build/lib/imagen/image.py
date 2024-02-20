"""
PatternGenerators based on bitmap images stored in files.

Requires the Python Imaging Library (PIL). In general, the pillow fork
of PIL is recommended as it is being actively maintained and works
with Python 3.
"""

# StringIO.StringIO is *not* the same as io.StringIO:
# https://mail.python.org/pipermail/python-list/2013-May/648080.html

# In short, the former accepts bytes whereas the latter only accepts
# unicode. In Python 3, BytesIO may be used with pillow safely.

try:
    from StringIO import StringIO as BytesIO
except:
    from io import BytesIO

from PIL import Image
from PIL import ImageOps

import numpy as np

import param
from param.parameterized import overridable_property
from holoviews.core import BoundingBox, SheetCoordinateSystem

from .patterngenerator import ChannelGenerator, ChannelTransform
from .transferfn import DivisiveNormalizeLinf, TransferFn

from os.path import splitext

import numbergen

class ImageSampler(param.Parameterized):
    """
    A class of objects that, when called, sample an image.
    """
    __abstract=True

    def _get_image(self):
        # CB: In general, might need to consider caching to avoid
        # loading of image/creation of scs and application of wpofs
        # every time/whatever the sampler does to set up the image
        # before sampling
        return self._image

    def _set_image(self,image):
        self._image = image

    def _del_image(self):
        del self._image

    # As noted by JP in FastImageSampler, this isn't easy to figure out.
    def __call__(self,image,x,y,sheet_xdensity,sheet_ydensity,width=1.0,height=1.0):
        raise NotImplementedError

    image = overridable_property(_get_image,_set_image,_del_image)



# CEBALERT: ArraySampler?
class PatternSampler(ImageSampler):
    """
    When called, resamples - according to the size_normalization
    parameter - an image at the supplied (x,y) sheet coordinates.

    (x,y) coordinates outside the image are returned as the background
    value.
    """
    whole_pattern_output_fns = param.HookList(class_=TransferFn,default=[],doc="""
        Functions to apply to the whole image before any sampling is done.""")

    background_value_fn = param.Callable(default=None,doc="""
        Function to compute an appropriate background value. Must accept
        an array and return a scalar.""")

    size_normalization = param.ObjectSelector(default='original',
        objects=['original','stretch_to_fit','fit_shortest','fit_longest'],
        doc="""
        Determines how the pattern is scaled initially, relative to
        the default retinal dimension of 1.0 in sheet coordinates:

        'stretch_to_fit': scale both dimensions of the pattern so they
        would fill a Sheet with bounds=BoundingBox(radius=0.5)
        (disregards the original's aspect ratio).

        'fit_shortest': scale the pattern so that its shortest
        dimension is made to fill the corresponding dimension on a
        Sheet with bounds=BoundingBox(radius=0.5) (maintains the
        original's aspect ratio, filling the entire bounding box).

        'fit_longest': scale the pattern so that its longest dimension
        is made to fill the corresponding dimension on a Sheet with
        bounds=BoundingBox(radius=0.5) (maintains the original's
        aspect ratio, fitting the image into the bounding box but not
        necessarily filling it).

        'original': no scaling is applied; each pixel of the pattern
        corresponds to one matrix unit of the Sheet on which the
        pattern being displayed.""")

    def _get_image(self):
        return self.scs.activity

    def _set_image(self,image):
        # Stores a SheetCoordinateSystem with an activity matrix
        # representing the image
        if not isinstance(image,np.ndarray):
            image = np.array(image,np.float)

        rows,cols = image.shape
        self.scs = SheetCoordinateSystem(xdensity=1.0,ydensity=1.0,
                                         bounds=BoundingBox(points=((-cols/2.0,-rows/2.0),
                                                                    ( cols/2.0, rows/2.0))))
        self.scs.activity=image

    def _del_image(self):
        self.scs = None


    def __call__(self, image, x, y, sheet_xdensity, sheet_ydensity, width=1.0, height=1.0):
        """
        Return pixels from the supplied image at the given Sheet (x,y)
        coordinates.

        The image is assumed to be a NumPy array or other object that
        exports the NumPy buffer interface (i.e. can be converted to a
        NumPy array by passing it to numpy.array(), e.g. Image.Image).
        The whole_pattern_output_fns are applied to the image before
        any sampling is done.

        To calculate the sample, the image is scaled according to the
        size_normalization parameter, and any supplied width and
        height. sheet_xdensity and sheet_ydensity are the xdensity and
        ydensity of the sheet on which the pattern is to be drawn.
        """
        # CEB: could allow image=None in args and have 'if image is
        # not None: self.image=image' here to avoid re-initializing the
        # image.
        self.image=image

        for wpof in self.whole_pattern_output_fns:
            wpof(self.image)
        if not self.background_value_fn:
            self.background_value = 0.0
        else:
            self.background_value = self.background_value_fn(self.image)

        pattern_rows,pattern_cols = self.image.shape

        if width==0 or height==0 or pattern_cols==0 or pattern_rows==0:
            return np.ones(x.shape)*self.background_value

        # scale the supplied coordinates to match the pattern being at density=1
        x=x*sheet_xdensity # deliberately don't operate in place (so as not to change supplied x & y)
        y=y*sheet_ydensity

        # scale according to initial pattern size_normalization selected (size_normalization)
        self.__apply_size_normalization(x,y,sheet_xdensity,sheet_ydensity,self.size_normalization)

        # scale according to user-specified width and height
        x/=width
        y/=height

        # now sample pattern at the (r,c) corresponding to the supplied (x,y)
        r,c = self.scs.sheet2matrixidx(x,y)
        # (where(cond,x,y) evaluates x whether cond is True or False)
        r.clip(0,pattern_rows-1,out=r)
        c.clip(0,pattern_cols-1,out=c)
        left,bottom,right,top = self.scs.bounds.lbrt()
        return np.where((x>=left) & (x<right) & (y>bottom) & (y<=top),
                           self.image[r,c],
                           self.background_value)


    def __apply_size_normalization(self,x,y,sheet_xdensity,sheet_ydensity,size_normalization):
        pattern_rows,pattern_cols = self.image.shape

        # Instead of an if-test, could have a class of this type of
        # function (c.f. OutputFunctions, etc)...
        if size_normalization=='original':
            return

        elif size_normalization=='stretch_to_fit':
            x_sf,y_sf = pattern_cols/sheet_xdensity, pattern_rows/sheet_ydensity
            x*=x_sf; y*=y_sf

        elif size_normalization=='fit_shortest':
            if pattern_rows<pattern_cols:
                sf = pattern_rows/sheet_ydensity
            else:
                sf = pattern_cols/sheet_xdensity
            x*=sf;y*=sf

        elif size_normalization=='fit_longest':
            if pattern_rows<pattern_cols:
                sf = pattern_cols/sheet_xdensity
            else:
                sf = pattern_rows/sheet_ydensity
            x*=sf;y*=sf



def edge_average(a):
    "Return the mean value around the edge of an array."

    if len(np.ravel(a)) < 2:
        return float(a[0])
    else:
        top_edge = a[0]
        bottom_edge = a[-1]
        left_edge = a[1:-1,0]
        right_edge = a[1:-1,-1]

        edge_sum = np.sum(top_edge) + np.sum(bottom_edge) + np.sum(left_edge) + np.sum(right_edge)
        num_values = len(top_edge)+len(bottom_edge)+len(left_edge)+len(right_edge)

        return float(edge_sum)/num_values



class FastImageSampler(ImageSampler):
    """
    A fast-n-dirty image sampler using Python Imaging Library
    routines.  Currently this sampler doesn't support user-specified
    size_normalization or cropping but rather simply scales and crops
    the image to fit the given matrix size without distorting the
    aspect ratio of the original picture.
    """

    sampling_method = param.Integer(default=Image.NEAREST,doc="""
       Python Imaging Library sampling method for resampling an image.
       Defaults to Image.NEAREST.""")

    def _set_image(self,image):
        if not isinstance(image,Image.Image):
            self._image = Image.new('L',image.shape)
            self._image.putdata(image.ravel())
        else:
            self._image = image

    def __call__(self, image, x, y, sheet_xdensity, sheet_ydensity, width=1.0, height=1.0):
        self.image=image

        # JPALERT: Right now this ignores all options and just fits the image into given array.
        # It needs to be fleshed out to properly size and crop the
        # image given the options. (maybe this class needs to be
        # redesigned?  The interface to this function is pretty inscrutable.)
        im = ImageOps.fit(self.image,x.shape,self.sampling_method)
        return np.array(im,dtype=np.float)



class GenericImage(ChannelGenerator):
    """
    Generic 2D image generator with support for multiple channels.

    Subclasses should override the _get_image method to produce the
    image object.

    By default, the background value is calculated as an edge average:
    see edge_average().  Black-bordered images therefore have a black
    background, and white-bordered images have a white
    background. Images with no border have a background that is less
    of a contrast than a white or black one.

    At present, rotation, size_normalization, etc. just resample; it
    would be nice to support some interpolation options as well.
    """

    __abstract = True

    aspect_ratio  = param.Number(default=1.0,bounds=(0.0,None),
        softbounds=(0.0,2.0),precedence=0.31,doc="""
        Ratio of width to height; size*aspect_ratio gives the width.""")

    size  = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.30,doc="""
        Height of the image.""")

    pattern_sampler = param.ClassSelector(class_=ImageSampler,
        default=PatternSampler(background_value_fn=edge_average,
                               size_normalization='fit_shortest',
                               whole_pattern_output_fns=[DivisiveNormalizeLinf()]),doc="""
        The PatternSampler to use to resample/resize the image.""")

    cache_image = param.Boolean(default=False,doc="""
        If False, discards the image and pattern_sampler after drawing
        the pattern each time, to make it possible to use very large
        databases of images without running out of memory.""")


    def __init__(self, **params):
        self._image = None
        super(GenericImage, self).__init__(**params)
        self._get_image(self)


    def _get_image(self,p):
        """
        If necessary as indicated by the parameters, get a new image,
        assign it to self._image and return True.  If no new image is
        needed, return False.
        """
        raise NotImplementedError


    def _reduced_call(self, **params_to_override):
        """
        Simplified version of PatternGenerator's __call__ method.
        """
        p=param.ParamOverrides(self,params_to_override)

        fn_result = self.function(p)
        self._apply_mask(p,fn_result)
        result = p.scale*fn_result+p.offset
        return result


    def _process_channels(self,p,**params_to_override):
        """
        Add the channel information to the channel_data attribute.
        """
        orig_image = self._image

        for i in range(len(self._channel_data)):
            self._image = self._original_channel_data[i]
            self._channel_data[i] = self._reduced_call(**params_to_override)
        self._image = orig_image
        return self._channel_data


    def function(self,p):
        height = p.size
        width = p.aspect_ratio*height

        result = p.pattern_sampler(self._get_image(p),p.pattern_x,p.pattern_y,
                                   float(p.xdensity),float(p.ydensity),
                                   float(width),float(height))
        if p.cache_image is False:
            self._image = None
            del self.pattern_sampler.image

        return result


    def __getstate__(self):
        """
        Return the object's state (as in the superclass), but replace
        the '_image' attribute's Image with a string representation.
        """
        state = super(GenericImage,self).__getstate__()

        if '_image' in state and state['_image'] is not None:
            f = BytesIO()
            image = state['_image']
            # format could be None (we should probably just not save in that case)
            image.save(f,format=image.format or 'TIFF')
            state['_image'] = f.getvalue()
            f.close()

        return state


    def __setstate__(self,state):
        """
        Load the object's state (as in the superclass), but replace
        the '_image' string with an actual Image object.
        """
        # state['_image'] is apparently sometimes None (see SF #2276819).
        if '_image' in state and state['_image'] is not None:
            state['_image'] = Image.open(BytesIO(state['_image']))
        super(GenericImage,self).__setstate__(state)



class FileImage(GenericImage):
    """
    2D Image generator that reads the image from a file.

    Grayscale versions of the image are always available, converted
    from the color version if necessary.  For color images,
    three-channel color values are available through the channels()
    method. See Image's Image class for details of supported image
    file formats.
    """

    filename = param.Filename(default='images/ellen_arthur.pgm', precedence=0.9,doc="""
        File path (can be relative to Param's base path) to a bitmap
        image.  The image can be in any format accepted by PIL,
        e.g. PNG, JPG, TIFF, or PGM as well or numpy save files (.npy
        or .npz) containing 2D or 3D arrays (where the third dimension
        is used for each channel).""")


    def __init__(self, **params):
        self.last_filename = None  # Cached to avoid unnecessary reloading for each channel
        self._cached_average = None
        super(FileImage,self).__init__(**params) ## must be called after setting the class-attributes
        self._image = None # necessary to ensure reloading of data (due to cache mechanisms
                           # call after super.__init__, which calls _get_image()


    def __call__(self,**params_to_override):
        # Cache image to avoid channel_data being deleted before channel-specific processing completes.
        p = param.ParamOverrides(self,params_to_override)

        if not (p.cache_image and (p._image is not None)):
            self._cached_average = super(FileImage,self).__call__(**params_to_override)

            self._channel_data = self._process_channels(p,**params_to_override)

            for c in self.channel_transforms:
                self._channel_data = c(self._channel_data)

            if p.cache_image is False:
                self._image = None


        return self._cached_average


    def set_matrix_dimensions(self, *args):
        """
        Subclassed to delete the cached image when matrix dimensions
        are changed.
        """
        self._image = None
        super(FileImage, self).set_matrix_dimensions(*args)


    def _get_image(self,p):
        file_, ext = splitext(p.filename)
        npy = (ext.lower() == ".npy")
        reload_image = (p.filename!=self.last_filename or self._image is None)

        self.last_filename = p.filename

        if reload_image:
            if npy:
                self._load_npy(p.filename)
            else:
                self._load_pil_image(p.filename)

        return self._image


    def _load_pil_image(self, filename):
        """
        Load image using PIL.
        """
        self._channel_data = []
        self._original_channel_data = []

        im = Image.open(filename)
        self._image = ImageOps.grayscale(im)
        im.load()

        file_data = np.asarray(im, float)
        file_data = file_data / file_data.max()

        # if the image has more than one channel, load them
        if( len(file_data.shape) == 3 ):
            num_channels = file_data.shape[2]
            for i in range(num_channels):
                self._channel_data.append( file_data[:, :, i])
                self._original_channel_data.append( file_data[:, :, i] )


    def _load_npy(self, filename):
        """
        Load image using Numpy.
        """
        self._channel_data = []
        self._original_channel_data = []
        file_channel_data = np.load(filename)
        file_channel_data = file_channel_data / file_channel_data.max()

        for i in range(file_channel_data.shape[2]):
            self._channel_data.append(file_channel_data[:, :, i])
            self._original_channel_data.append(file_channel_data[:, :, i])

        self._image = file_channel_data.sum(2) / file_channel_data.shape[2]




class RotateHue(ChannelTransform):
    """
    Rotate the hue of an Image PatternGenerator.

    Requires a three-channel (e.g. RGB) or a 4-channel (e.g. RGBA)
    color image.  Also allows the saturation of the image to be
    scaled.

    Requires the color space of the image to be declared using the
    colorspaces.color_conversion object, and uses the analysis color
    space from that object to do the rotation.
    """

    saturation = param.Number(default=1.0,doc="""
        Scale the saturation by the specified value.""")

    rotation = param.Number(default=numbergen.UniformRandom(
        name='hue_jitter', lbound=0,ubound=1, seed=1048921),
                            softbounds=(0.0,1.0),doc="""
        Amount by which to rotate the hue.  The default setting
        chooses a random value of hue rotation between zero and 100%.
        If set to 0, no rotation will be performed.""")


    def __call__(self,channel_data):
        assert( len(channel_data)==3 or len(channel_data)==4 )

        from .colorspaces import color_conversion as cc

        channs_in  = np.dstack(channel_data[0:3])
        channs_out = cc.image2working(channs_in)
        analysis_space = cc.working2analysis(channs_out)

        if self.rotation != 0:
            cc.jitter_hue(analysis_space,self.rotation)
        cc.multiply_sat(analysis_space,self.saturation)

        channs_out = cc.analysis2working(analysis_space)

        # Takes only the first three channels (e.g. RGB)
        channel_data[0:3] = np.dsplit(channs_out, 3)
        for a in channel_data:
            a.shape = a.shape[0:2]

        return channel_data



class ScaleChannels(ChannelTransform):
    """
    Scale each channel of an Image PatternGenerator by a different
    factor.

    The list of channel factors should be the same length as the
    number of channels.  Otherwise, if the factors provided are fewer
    than the channels of the Image, the remaining channels will not be
    scaled. If they are more, then only the first N factors are used.
    """

    channel_factors = param.Dynamic(default=[1.0,1.0,1.0],doc="""
        Channel scaling factors.""")


    def __call__(self,channel_data):
        # safety check
        num_channels = min( len(channel_data), len(self.channel_factors) )
        for i in range( num_channels ):
            #TFALERT: Not sure why this is required, it should work out of the box
            #Maybe because channel_factors should be a param.List rather than
            #param.Dynamic?
            if(callable(self.channel_factors[i])):
                channel_data[i] = channel_data[i] * self.channel_factors[i]()
            else:
                channel_data[i] = channel_data[i] * self.channel_factors[i]
            channel_data[i][channel_data[i]>1]=1.0

        return channel_data
