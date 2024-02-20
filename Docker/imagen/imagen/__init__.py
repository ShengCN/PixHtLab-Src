"""
Objects capable of generating a two-dimensional array of values.

Such patterns can be used as input to machine learning, neural
network, or compuatational neuroscience algorithms, or for any other
purpose where a two-dimensional pattern may be needed.  Any new
PatternGenerator classes can be derived from these, and can then be
combined with the existing classes easily.
"""

import sys, os, copy

# Add param submodule to sys.path
cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..', 'param'))
sys.path.insert(0, os.path.join(cwd, '..', 'holoviews'))

import param
from param.version import Version

__version__ = Version(release=(2,1,0), fpath=__file__,
                      commit="$Format:%h$", reponame='imagen')


import numpy as np
from numpy import pi

from param.parameterized import ParamOverrides
from param import ClassSelector

# Imported here so that all PatternGenerators will be in the same package
from .patterngenerator import PatternGenerator, CompositeBase, Composite
from .patterngenerator import Constant, ChannelTransform, ChannelGenerator # pyflakes:ignore (API import)
from .patterngenerator import CorrelateChannels, ComposeChannels # pyflakes:ignore (API import)


from holoviews.element import Image                    # pyflakes:ignore (API import)

from holoviews.core import SheetCoordinateSystem       # pyflakes:ignore (API import)
from holoviews.core import boundingregion, sheetcoords # pyflakes:ignore (API import)

from .patternfn import gaussian,exponential,gabor,line,disk,ring,\
    sigmoid,arc_by_radian,arc_by_center,smooth_rectangle,float_error_ignore, \
    log_gaussian

import numbergen
from imagen.transferfn import DivisiveNormalizeL1

# Could add a Gradient class, where the brightness varies as a
# function of an equation for a plane.  This could be useful as a
# background, or to see how sharp a gradient is needed to get a
# response.


class HalfPlane(PatternGenerator):
    """
    Constant pattern on in half of the plane, and off in the rest,
    with optional Gaussian smoothing.
    """

    smoothing = param.Number(default=0.02,bounds=(0.0,None),softbounds=(0.0,0.5),
                             precedence=0.61,doc="Width of the Gaussian fall-off.")

    def function(self,p):
        if p.smoothing==0.0:
            falloff=self.pattern_y*0.0
        else:
            with float_error_ignore():
                falloff=np.exp(np.divide(-self.pattern_y*self.pattern_y,
                                                2*p.smoothing*p.smoothing))

        return np.where(self.pattern_y>0.0,1.0,falloff)


class Gaussian(PatternGenerator):
    """
    2D Gaussian pattern generator.

    The sigmas of the Gaussian are calculated from the size and
    aspect_ratio parameters:

      ysigma=size/2
      xsigma=ysigma*aspect_ratio

    The Gaussian is then computed for the given (x,y) values as::

      exp(-x^2/(2*xsigma^2) - y^2/(2*ysigma^2)
    """

    aspect_ratio = param.Number(default=1/0.31,bounds=(0.0,None),softbounds=(0.0,6.0),
        precedence=0.31,doc="""
        Ratio of the width to the height.
        Specifically, xsigma=ysigma*aspect_ratio (see size).""")

    size = param.Number(default=0.155,doc="""
        Overall size of the Gaussian, defined by:
        exp(-x^2/(2*xsigma^2) - y^2/(2*ysigma^2)
        where ysigma=size/2 and xsigma=size/2*aspect_ratio.""")

    def function(self,p):
        ysigma = p.size/2.0
        xsigma = p.aspect_ratio*ysigma

        return gaussian(self.pattern_x,self.pattern_y,xsigma,ysigma)


class ExponentialDecay(PatternGenerator):
    """
    2D Exponential pattern generator.

    Exponential decay based on distance from a central peak,
    i.e. exp(-d), where d is the distance from the center (assuming
    size=1.0 and aspect_ratio==1.0).  More generally, the size and
    aspect ratio determine the scaling of x and y dimensions:

      yscale=size/2
      xscale=yscale*aspect_ratio

    The exponential is then computed for the given (x,y) values as::

      exp(-sqrt((x/xscale)^2 - (y/yscale)^2))
    """

    aspect_ratio = param.Number(default=1/0.31,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="""Ratio of the width to the height.""")

    size = param.Number(default=0.155,doc="""
        Overall scaling of the x and y dimensions.""")

    def function(self,p):
        yscale = p.size/2.0
        xscale = p.aspect_ratio*yscale

        return exponential(self.pattern_x,self.pattern_y,xscale,yscale)


class SineGrating(PatternGenerator):
    """2D sine grating pattern generator."""

    frequency = param.Number(default=2.4,bounds=(0.0,None),softbounds=(0.0,10.0),
                       precedence=0.50, doc="Frequency of the sine grating.")

    phase     = param.Number(default=0.0,bounds=(0.0,None),softbounds=(0.0,2*pi),
                       precedence=0.51,doc="Phase of the sine grating.")

    def function(self,p):
        """Return a sine grating pattern (two-dimensional sine wave)."""
        return 0.5 + 0.5*np.sin(p.frequency*2*pi*self.pattern_y + p.phase)



class Gabor(PatternGenerator):
    """2D Gabor pattern generator."""

    frequency = param.Number(default=2.4,bounds=(0.0,None),softbounds=(0.0,10.0),
        precedence=0.50,doc="Frequency of the sine grating component.")

    phase = param.Number(default=0.0,bounds=(0.0,None),softbounds=(0.0,2*pi),
        precedence=0.51,doc="Phase of the sine grating component.")

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc=
        """
        Ratio of pattern width to height.
        The width of the Gaussian component is size*aspect_ratio (see Gaussian).
        """)

    size = param.Number(default=0.25,doc="""
        Determines the height of the Gaussian component (see Gaussian).""")

    def function(self,p):
        height = p.size/2.0
        width = p.aspect_ratio*height

        return gabor(self.pattern_x,self.pattern_y,width,height,
                     p.frequency,p.phase)


class Line(PatternGenerator):
    """2D line pattern generator."""

    # Hide unused parameters
    size = param.Number(precedence=-1.0)

    thickness = param.Number(default=0.006,bounds=(0.0,None),softbounds=(0.0,1.0),
                             precedence=0.60,doc="""
        Thickness (width) of the solid central part of the line.""")

    enforce_minimal_thickness = param.Boolean(default=False,precedence=0.60, doc="""
        If True, ensure that the line is at least one pixel in width even for
        small thicknesses where the line could otherwise fall in between pixel
        centers and thus disappear at some orientations.""")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
                             precedence=0.61, doc="""
        Width of the Gaussian fall-off.""")


    def _pixelsize(self, p):
        """Calculate line width necessary to cover at least one pixel on all axes."""
        xpixelsize = 1./float(p.xdensity)
        ypixelsize = 1./float(p.ydensity)
        return max([xpixelsize,ypixelsize])

    def _effective_thickness(self, p):
        """Enforce minimum thickness based on the minimum pixel size."""
        return max([p.thickness,self._pixelsize(p)])

    def _count_pixels_on_line(self, y, p):
        """Count the number of pixels rendered on this line."""
        h = line(y, self._effective_thickness(p), 0.0)
        return h.sum()

    def _minimal_y(self, p):
        """
        For the specified y and one offset by half a pixel, return the
        one that results in the fewest pixels turned on, so that when
        the thickness has been enforced to be at least one pixel, no
        extra pixels are needlessly included (which would cause
        double-width lines).
        """
        y0 = self.pattern_y
        y1 = y0 + self._pixelsize(p)/2.
        return y0 if self._count_pixels_on_line(y0, p) < self._count_pixels_on_line(y1, p) else y1

    def function(self,p):
        return line(
            self.pattern_y if not p.enforce_minimal_thickness else self._minimal_y(p),
            p.thickness    if not p.enforce_minimal_thickness else self._effective_thickness(p),
            p.smoothing)



class Disk(PatternGenerator):
    """
    2D disk pattern generator.

    An elliptical disk can be obtained by adjusting the aspect_ratio
    of a circular disk; this transforms a circle into an ellipse by
    stretching the circle in the y (vertical) direction.

    The Gaussian fall-off at a point P is an approximation for
    non-circular disks, since the point on the ellipse closest to P is
    taken to be the same point as the point on the circle before
    stretching that was closest to P.
    """

    aspect_ratio  = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc=
        "Ratio of width to height; size*aspect_ratio gives the width of the disk.")

    size  = param.Number(default=0.5,doc="Top to bottom height of the disk")

    smoothing = param.Number(default=0.1,bounds=(0.0,None),softbounds=(0.0,0.5),
                       precedence=0.61,doc="Width of the Gaussian fall-off")

    def function(self,p):
        height = p.size

        if p.aspect_ratio==0.0:
            return self.pattern_x*0.0

        return disk(self.pattern_x/p.aspect_ratio,self.pattern_y,height,
                    p.smoothing)


class Ring(PatternGenerator):
    """
    2D ring pattern generator.

    See the Disk class for a note about the Gaussian fall-off.
    """

    thickness = param.Number(default=0.015,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the ring.")

    smoothing = param.Number(default=0.1,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the ring.")

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc=
        "Ratio of width to height; size*aspect_ratio gives the overall width.")

    size = param.Number(default=0.5)

    def function(self,p):
        height = p.size
        if p.aspect_ratio==0.0:
            return self.pattern_x*0.0

        return ring(self.pattern_x/p.aspect_ratio,self.pattern_y,height,
                    p.thickness,p.smoothing)


class OrientationContrast(SineGrating):
    """
    Circular pattern for testing responses to differences in contrast.

    The pattern contains a sine grating ring surrounding a sine
    grating disk, each with parameters (orientation, size, scale and
    offset) that can be changed independently.
    """

    orientationcenter   = param.Number(default=0.0,bounds=(0.0,2*pi), doc="Orientation of the center grating.")
    orientationsurround = param.Number(default=0.0,bounds=(-pi*2,pi*2), doc="Orientation of the surround grating, either absolute or relative to the central grating.")
    surround_orientation_relative = param.Boolean(default=False, doc="Determines whether the surround grating is relative to the central grating.")
    sizecenter     = param.Number(default=0.5,bounds=(0.0,None),softbounds=(0.0,10.0), doc="Size of the center grating.")
    sizesurround   = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,10.0), doc="Size of the surround grating.")
    scalecenter    = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,10.0), doc="Scale of the center grating.")
    scalesurround  = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,10.0), doc="Scale of the surround grating.")
    offsetcenter   = param.Number(default=0.0,bounds=(0.0,None),softbounds=(0.0,10.0), doc="Offset of the center grating.")
    offsetsurround = param.Number(default=0.0,bounds=(0.0,None),softbounds=(0.0,10.0), doc="Offset of the surround grating.")
    smoothing      = param.Number(default=0.0,bounds=(0.0,None),softbounds=(0.0,0.5),  doc="Width of the Gaussian fall-off inside and outside the ring.")
    thickness      = param.Number(default=0.3,bounds=(0.0,None),softbounds=(0.0,0.5),  doc="Thickness (line width) of the ring.")
    aspect_ratio   = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),  doc="Ratio of width to height; size*aspect_ratio gives the overall width.")
    size           = param.Number(default=0.5)

    def __call__(self,**params_to_override):
        p = ParamOverrides(self,params_to_override)
        input_1=SineGrating(mask_shape=Disk(smoothing=0,size=1.0),phase=p.phase, frequency=p.frequency,
                            orientation=p.orientationcenter,
                            scale=p.scalecenter, offset=p.offsetcenter,
                            x=p.x, y=p.y,size=p.sizecenter)
        if p.surround_orientation_relative:
            surround_or = p.orientationcenter + p.orientationsurround
        else:
            surround_or = p.orientationsurround
        input_2=SineGrating(mask_shape=Ring(thickness=p.thickness,smoothing=0,size=1.0),phase=p.phase, frequency=p.frequency,
                            orientation=surround_or, scale=p.scalesurround, offset=p.offsetsurround,
                            x=p.x, y=p.y, size=p.sizesurround)

        patterns = [input_1(xdensity=p.xdensity,ydensity=p.ydensity,bounds=p.bounds),
                    input_2(xdensity=p.xdensity,ydensity=p.ydensity,bounds=p.bounds)]

        image_array = np.add.reduce(patterns)
        return image_array



class RawRectangle(PatternGenerator):
    """
    2D rectangle pattern generator with no smoothing, for use when
    drawing patterns pixel by pixel.
    """

    aspect_ratio   = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc=
        "Ratio of width to height; size*aspect_ratio gives the width of the rectangle.")

    size  = param.Number(default=0.5,doc="Height of the rectangle.")

    def function(self,p):
        height = p.size
        width = p.aspect_ratio*height
        return np.bitwise_and(np.abs(self.pattern_x)<=width/2.0,
                           np.abs(self.pattern_y)<=height/2.0)



class Rectangle(PatternGenerator):
    """
    2D rectangle pattern, with Gaussian smoothing around the
    edges.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,6.0),
        precedence=0.31,doc=
        "Ratio of width to height; size*aspect_ratio gives the width of the rectangle.")

    size = param.Number(default=0.5,doc="Height of the rectangle.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off outside the rectangle.")

    def function(self,p):
        height=p.size
        width=p.aspect_ratio*height

        return smooth_rectangle(self.pattern_x, self.pattern_y,
                                width, height, p.smoothing, p.smoothing)



class Arc(PatternGenerator):
    """
    2D arc pattern generator.

    Draws an arc (partial ring) of the specified size (radius*2),
    starting at radian 0.0 and ending at arc_length.  The orientation
    can be changed to choose other start locations.  The pattern is
    centered at the center of the ring.

    See the Disk class for a note about the Gaussian fall-off.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,6.0),
        precedence=0.31,doc="""
        Ratio of width to height; size*aspect_ratio gives the overall width.""")

    thickness = param.Number(default=0.015,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the ring.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the ring.")

    arc_length = param.Number(default=pi,bounds=(0.0,None),softbounds=(0.0,2.0*pi),
                              inclusive_bounds=(True,False),precedence=0.62, doc="""
        Length of the arc, in radians, starting from orientation 0.0.""")

    size = param.Number(default=0.5)

    def function(self,p):
        if p.aspect_ratio==0.0:
            return self.pattern_x*0.0

        return arc_by_radian(self.pattern_x/p.aspect_ratio, self.pattern_y, p.size,
                             (2*pi-p.arc_length, 0.0), p.thickness, p.smoothing)


class Curve(Arc):
    """
    2D curve pattern generator.

    Based on Arc, but centered on a tangent point midway through the
    arc, rather than at the center of a ring, and with curvature
    controlled directly rather than through the overall size of the
    pattern.

    Depending on the size_type, the size parameter can control either
    the width of the pattern, keeping this constant regardless of
    curvature, or the length of the curve, keeping that constant
    instead (as for a long thin object being bent).

    Specifically, for size_type=='constant_length', the curvature
    parameter determines the ratio of height to width of the arc, with
    positive curvature for concave shape and negative for convex. The
    size parameter determines the width of the curve.

    For size_type=='constant_width', the curvature parameter
    determines the portion of curve radian to 2pi, and the curve
    radius is changed accordingly following the formula::

      size=2pi*radius*curvature

    Thus, the size parameter determines the total length of the
    curve. Positive curvature stands for concave shape, and negative
    for convex.

    See the Disk class for a note about the Gaussian fall-off.
    """

    # Hide unused parameters
    arc_length = param.Number(precedence=-1.0)
    aspect_ratio = param.Number(default=1.0, precedence=-1.0)

    size_type = param.ObjectSelector(default='constant_length',
        objects=['constant_length','constant_width'],precedence=0.61,doc="""
        For a given size, whether to draw a curve with that total length,
        or with that width, keeping it constant as curvature is varied.""")

    curvature = param.Number(default=0.5, bounds=(-0.5, 0.5), precedence=0.62, doc="""
        Ratio of height to width of the arc, with positive value giving
        a concave shape and negative value giving convex.""")

    def function(self,p):
        return arc_by_center(self.pattern_x/p.aspect_ratio,self.pattern_y,
                             (p.size,p.size*p.curvature),
                             (p.size_type=='constant_length'),
                             p.thickness, p.smoothing)



class SquareGrating(PatternGenerator):
    """2D squarewave (symmetric or asymmetric) grating pattern generator."""

    frequency = param.Number(default=2.4,bounds=(0.0,None),softbounds=(0.0,10.0),
        precedence=0.50,doc="Frequency of the square grating.")

    phase     = param.Number(default=0.0,bounds=(0.0,None),softbounds=(0.0,2*pi),
        precedence=0.51,doc="Phase of the square grating.")

    duty_cycle = param.Number(default=0.5,bounds=(0.0,1.0),
        precedence=0.51,doc="""
        The duty cycle is the ratio between the pulse duration (width of the bright bar)
        and the period (1/frequency).
        The pulse is defined as the time during which the square wave signal is 1 (high).""")

    # We will probably want to add anti-aliasing to this,
    # and there might be an easier way to do it than by
    # cropping a sine grating.

    def function(self,p):
        """
        Return a square-wave grating (alternating black and white bars).
        """
        return np.around(
            0.5 +
            0.5*np.sin(pi*(p.duty_cycle-0.5)) +
            0.5*np.sin(p.frequency*2*pi*self.pattern_y + p.phase))


#JABALERT: replace with x%1.0 below
def wrap(lower, upper, x):
    """
    Circularly alias the numeric value x into the range [lower,upper).

    Valid for cyclic quantities like orientations or hues.
    """
    #I have no idea how I came up with this algorithm; it should be simplified.
    #
    # Note that Python's % operator works on floats and arrays;
    # usually one can simply use that instead.  E.g. to wrap array or
    # scalar x into 0,2*pi, just use "x % (2*pi)".
    range_=upper-lower
    return lower + np.fmod(x-lower + 2*range_*(1-np.floor(x/(2*range_))), range_)



class Selector(CompositeBase):
    """
    PatternGenerator that selects from a list of other PatternGenerators.
    """

    # CB: needs to have time_fn=None
    index = param.Number(default=numbergen.UniformRandom(lbound=0,ubound=1.0,seed=76),
        bounds=(-1.0,1.0),precedence=0.20,doc="""
        Index into the list of pattern generators, on a scale from 0
        (start of the list) to 1.0 (end of the list).  Typically a
        random value or other number generator, to allow a different item
        to be selected each time.""")


    def function(self,p):
        """Selects and returns one of the patterns in the list."""
        int_index=int(len(p.generators)*wrap(0,1.0,p.index))
        pg=p.generators[int_index]

        image_array = pg(xdensity=p.xdensity,ydensity=p.ydensity,bounds=p.bounds,
                         x=p.x+p.size*(pg.x*np.cos(p.orientation)-pg.y*np.sin(p.orientation)),
                         y=p.y+p.size*(pg.x*np.sin(p.orientation)+pg.y*np.cos(p.orientation)),
                         orientation=pg.orientation+p.orientation,size=pg.size*p.size,
                         scale=pg.scale*p.scale,offset=pg.offset+p.offset)

        return image_array

    def get_current_generator(self):
        """Return the current generator (as specified by self.index)."""
        int_index=int(len(self.generators)*wrap(0,1.0,self.inspect_value('index')))
        return self.generators[int_index]

    def channels(self, use_cached=False, **params_to_override):
        """
        Get channel data from the current generator.  use_cached is
        not supported at the moment, though it must be forced to be
        True in the current_generator in order to avoid generating the
        same data twice (the first time by self() and the second with
        current_generator.channels() ).
        """
        default = self(**params_to_override)
        current_generator = self.get_current_generator()

        res = current_generator.channels(use_cached=True)
        res['default'] = default

        return res

    def num_channels(self):
        """
        Get the number of channels in the input generators.
        """
        if(self.inspect_value('index') is None):
            if(len(self.generators)>0):
                return self.generators[0].num_channels()
            return 0

        return self.get_current_generator().num_channels()


class OffsetTimeFn(param.Parameterized):
    """
    A picklable version of the global time function with a custom offset
    and reset period.
    """

    offset = param.Number(default=0, doc="""
      The time offset from which frames are generated given the
      supplied pattern.""")

    reset_period = param.Number(default=4,bounds=(0,None),doc="""
        Period between generating each new translation episode.""")

    time_fn = param.Callable(default=param.Dynamic.time_fn,doc="""
        Function to generate the time used as a base for translation.""")

    def __call__(self):
        time = self.time_fn()
        return self.time_fn.time_type((time // self.reset_period) + self.offset)



class Sweeper(ChannelGenerator):
    """
    PatternGenerator that sweeps a supplied PatternGenerator in a
    direction perpendicular to its orientation. Each time step, the
    supplied PatternGenerator is sweeped further at a fixed speed, and
    after reset_period time steps a new pattern is drawn.
    """
    generator = param.ClassSelector(PatternGenerator,default=Gaussian(),precedence=0.97,
                                    doc="Pattern to sweep.")

    time_offset = param.Number(default=0, doc="""
      The time offset from which frames are generated given the
      supplied pattern.""")

    step_offset = param.Number(default=0, doc="""
      The number of steps to offset the sweeper by.""")

    reset_period = param.Number(default=4,bounds=(0,None),doc="""
        Period between generating each new translation episode.""")

    speed = param.Number(default=2.0/24.0,bounds=(0.0,None),doc="""
        The speed with which the pattern should move,
        in sheet coordinates per time_fn unit.""")

    relative_motion_orientation = param.Number(default=pi/2.0,bounds=(0,2*pi),doc="""
        The direction in which the pattern should be moved, relative
        to the orientation of the supplied generator""")

    time_fn = param.Callable(default=param.Dynamic.time_fn,doc="""
        Function to generate the time used as a base for translation.""")


    def num_channels(self):
        return self.generator.num_channels()


    def function(self, p):
        motion_time_fn = OffsetTimeFn(offset=p.time_offset,
                                      reset_period=p.reset_period,
                                      time_fn=p.time_fn)
        pg = p.generator
        pg.set_dynamic_time_fn(motion_time_fn)
        motion_orientation = pg.orientation + p.relative_motion_orientation

        step = int(p.time_fn() % p.reset_period) + p.step_offset

        new_x = p.x + p.size * pg.x
        new_y = p.y + p.size * pg.y

        try:
            #TFALERT: Not sure whether this is needed
            if(len(self._channel_data)!=len(pg._channel_data)):
               self._channel_data=copy.deepcopy(pg._channel_data)

            # For multichannel pattern generators
            for i in range(len(pg._channel_data)):
                self._channel_data[i] = pg.channels(
                    x=new_x + p.speed * step * np.cos(motion_orientation),
                    y=new_y + p.speed * step * np.sin(motion_orientation),
                    xdensity=p.xdensity, ydensity=p.ydensity,
                    bounds=p.bounds,
                    orientation=pg.orientation + p.orientation,
                    scale=pg.scale * p.scale, offset=pg.offset + p.offset)[i]
        except AttributeError:
            pass

        image_array = pg(xdensity=p.xdensity, ydensity=p.ydensity,
                         bounds=p.bounds,
                         x=new_x + p.speed * step * np.cos(motion_orientation),
                         y=new_y + p.speed * step * np.sin(motion_orientation),
                         orientation=pg.orientation + p.orientation,
                         scale=pg.scale * p.scale, offset=pg.offset + p.offset)

        return image_array



class Spiral(PatternGenerator):
    """
    Archimedean spiral.
    Successive turnings of the spiral have a constant separation distance.

    Spiral is defined by polar equation r=size*angle plotted in Gaussian plane.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    thickness = param.Number(default=0.02,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the spiral.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the spiral.")

    turning = param.Number(default=0.05,bounds=(0.01,None),softbounds=(0.01,2.0),
        precedence=0.62,doc="Density of turnings; turning*angle gives the actual radius.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = p.thickness
        gaussian_width = p.smoothing
        turning = p.turning

        spacing = turning*2*pi

        distance_from_origin = np.sqrt(x**2+y**2)
        distance_from_spiral_middle = np.fmod(spacing + distance_from_origin - turning*np.arctan2(y,x),spacing)

        distance_from_spiral_middle = np.minimum(distance_from_spiral_middle,spacing - distance_from_spiral_middle)
        distance_from_spiral = distance_from_spiral_middle - thickness/2.0

        spiral = 1.0 - np.greater_equal(distance_from_spiral,0.0)

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = np.exp(np.divide(-distance_from_spiral*distance_from_spiral, 2.0*sigmasq))

        return np.maximum(falloff, spiral)



class SpiralGrating(Composite):
    """
    Grating pattern made from overlaid spirals.
    """

    parts = param.Integer(default=2,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    thickness = param.Number(default=0.00,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the spiral.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the spiral.")

    turning = param.Number(default=0.05,bounds=(0.01,None),softbounds=(0.01,2.0),
        precedence=0.62,doc="Density of turnings; turning*angle gives the actual radius.")


    def function(self, p):
        gens = [Spiral(turning=p.turning,smoothing=p.smoothing,thickness=p.thickness,
                       orientation=i*2*np.pi/p.parts) for i in range(p.parts)]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()



class HyperbolicGrating(PatternGenerator):
    """
    Concentric rectangular hyperbolas with Gaussian fall-off which share the same asymptotes.
    abs(x^2/a^2 - y^2/a^2) = 1, where a mod size = 0
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    thickness = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness of the hyperbolas.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the hyperbolas.")

    size = param.Number(default=0.5,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.62,doc="Size as distance of inner hyperbola vertices from the centre.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = p.thickness
        gaussian_width = p.smoothing
        size = p.size

        distance_from_vertex_middle = np.fmod(np.sqrt(np.absolute(x**2 - y**2)),size)
        distance_from_vertex_middle = np.minimum(distance_from_vertex_middle,size - distance_from_vertex_middle)

        distance_from_vertex = distance_from_vertex_middle - thickness/2.0

        hyperbola = 1.0 - np.greater_equal(distance_from_vertex,0.0)

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = np.exp(np.divide(-distance_from_vertex*distance_from_vertex, 2.0*sigmasq))

        return np.maximum(falloff, hyperbola)



class Wedge(PatternGenerator):
    """
    A sector of a circle with Gaussian fall-off, with size determining the arc length.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    size = param.Number(default=pi/4,bounds=(0.0,None),softbounds=(0.0,2.0*pi),
        precedence=0.60,doc="Angular length of the sector, in radians.")

    smoothing = param.Number(default=0.4,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off outside the sector.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        gaussian_width = p.smoothing

        angle = np.absolute(np.arctan2(y,x))
        half_length = p.size/2

        radius = 1.0 - np.greater_equal(angle,half_length)
        distance = angle - half_length

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = np.exp(np.divide(-distance*distance, 2.0*sigmasq))

        return np.maximum(radius, falloff)



class RadialGrating(Composite):
    """
    Grating pattern made from alternating smooth circular segments (pie-shapes).
    """

    parts = param.Integer(default=4,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    smoothing = param.Number(default=0.8,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="""
        Width of the Gaussian fall-off outside the sector, scaled by parts.""")

    def function(self, p):
        gens = [Wedge(size=1.0/p.parts,smoothing=p.smoothing/p.parts,
                      orientation=i*2*np.pi/p.parts) for i in range(p.parts)]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()


class Asterisk(Composite):
    """
    Asterisk-like object composed of radial rectangular lines.
    Also makes crosses and tripods.
    """

    parts = param.Integer(default=3,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the asterisk.")

    thickness = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness of the rectangle.")

    smoothing = param.Number(default=0.015,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off around the rectangles.")

    size = param.Number(default=0.5,bounds=(0.01,None),softbounds=(0.1,2.0),
        precedence=0.62,doc="Overall diameter of the pattern.")

    def function(self, p):
        o=2*np.pi/p.parts
        gens = [Rectangle(orientation=i*o,smoothing=p.smoothing,
                          aspect_ratio=2*p.thickness/p.size,
                          size=p.size/2,
                          x=-p.size/4*np.sin(i*o),
                          y= p.size/4*np.cos(i*o))
                   for i in range(p.parts)]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()



class Angle(Composite):
    """
    Angle composed of two line segments.
    """

    thickness = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness of the rectangle.")

    smoothing = param.Number(default=0.015,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off around the rectangles.")

    size = param.Number(default=0.5,bounds=(0.01,None),softbounds=(0.1,2.0),
        precedence=0.62,doc="Overall diameter of the pattern, if angle=pi.")

    angle = param.Number(default=pi/4,bounds=(0.0,None),softbounds=(0,pi),
        precedence=0.63,doc="Angle between the two line segments.")

    def function(self, p):
        gens=[Rectangle(orientation=i*p.angle,smoothing=p.smoothing,
                        aspect_ratio=p.thickness/p.size,size=p.size,
                        x=-p.size/2*np.sin(i*p.angle))
              for i in [-1,1]]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()



class ConcentricRings(PatternGenerator):
    """
    Concentric rings with linearly increasing radius.
    Gaussian fall-off at the edges.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    thickness = param.Number(default=0.04,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the ring.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the rings.")

    size = param.Number(default=0.4,bounds=(0.01,None),softbounds=(0.1,2.0),
        precedence=0.62,doc="Radius difference of neighbouring rings.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = p.thickness
        gaussian_width = p.smoothing
        size = p.size

        distance_from_origin = np.sqrt(x**2+y**2)

        distance_from_ring_middle = np.fmod(distance_from_origin,size)
        distance_from_ring_middle = np.minimum(distance_from_ring_middle,size - distance_from_ring_middle)

        distance_from_ring = distance_from_ring_middle - thickness/2.0

        ring = 1.0 - np.greater_equal(distance_from_ring,0.0)

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = np.exp(np.divide(-distance_from_ring*distance_from_ring, 2.0*sigmasq))

        return np.maximum(falloff, ring)



class ArcCentered(Arc):
    """
    2D arc pattern generator (centered at the middle of the arc).

    Draws an arc (partial ring) of the specified size (radius*2),
    with middle at radian 0.0 and starting at arc_length/2 and ending
    at -arc_length/2. The pattern is centered at the middle of the arc.

    See the Disk class for a note about the Gaussian fall-off.
    """

    def function(self,p):
        if p.aspect_ratio==0.0:
            return self.pattern_x*0.0
        self.pattern_x -= (1+np.cos(pi-p.arc_length/2))*p.size/4

        return arc_by_radian((self.pattern_x+p.size/2)/p.aspect_ratio, self.pattern_y, p.size,
                             (2*pi-p.arc_length/2, p.arc_length/2), p.thickness, p.smoothing)


class DifferenceOfGaussians(PatternGenerator):
    """
    Two-dimensional difference of Gaussians pattern.
    """

    positive_size = param.Number(default=0.1, bounds=(0.0,None), softbounds=(0.0,5.0), precedence=(1),
        doc="""Size of the positive region of the pattern.""")

    positive_aspect_ratio = param.Number(default=1.5, bounds=(0.0,None), softbounds=(0.0,5.0), precedence=(2),
        doc="""Ratio of width to height for the positive region of the pattern.""")

    positive_x = param.Number(default=0.0, bounds=(None,None), softbounds=(-2.0,2.0), precedence=(3),
        doc="""X position for the central peak of the positive region.""")

    positive_y = param.Number(default=0.0, bounds=(None,None), softbounds=(-2.0,2.0), precedence=(4),
        doc="""Y position for the central peak of the positive region.""")


    negative_size = param.Number(default=0.3, bounds=(0.0,None), softbounds=(0.0,5.0), precedence=(5),
        doc="""Size of the negative region of the pattern.""")

    negative_aspect_ratio = param.Number(default=1.5, bounds=(0.0,None), softbounds=(0.0,5.0), precedence=(6),
        doc="""Ratio of width to height for the negative region of the pattern.""")

    negative_x = param.Number(default=0.0, bounds=(None,None), softbounds=(-2.0,2.0), precedence=(7),
        doc="""X position for the central peak of the negative region.""")

    negative_y = param.Number(default=0.0, bounds=(None,None), softbounds=(-2.0,2.0), precedence=(8),
        doc="""Y position for the central peak of the negative region.""")


    def function(self, p):
        positive = Gaussian(x=p.positive_x+p.x, y=p.positive_y+p.y,
            size=p.positive_size*p.size, aspect_ratio=p.positive_aspect_ratio,
            orientation=p.orientation, output_fns=[DivisiveNormalizeL1()])

        negative = Gaussian(x=p.negative_x+p.x, y=p.negative_y+p.y,
            size=p.negative_size*p.size, aspect_ratio=p.negative_aspect_ratio,
            orientation=p.orientation, output_fns=[DivisiveNormalizeL1()])

        return Composite(generators=[positive,negative], operator=np.subtract,
            xdensity=p.xdensity, ydensity=p.ydensity, bounds=p.bounds)()



class Sigmoid(PatternGenerator):
    """
    Two-dimensional sigmoid pattern, dividing the plane into positive
    and negative halves with a smoothly sloping transition between
    them.
    """

    slope = param.Number(default=10.0, bounds=(None,None), softbounds=(-100.0,100.0),
        doc="""Parameter controlling the smoothness of the transition
        between the two regions; high values give a sharp transition.""")


    def function(self, p):
        return sigmoid(self.pattern_y, p.slope)



class SigmoidedDoG(PatternGenerator):
    """
    Sigmoid multiplicatively combined with a difference of Gaussians,
    such that one part of the plane can be the mirror image of the other.
    """

    size = param.Number(default=0.5)

    positive_size = param.Number(default=0.15, bounds=(0.0,None), softbounds=(0.0,5.0), precedence=(1),
        doc="""Size of the positive Gaussian pattern.""")

    positive_aspect_ratio = param.Number(default=2.0, bounds=(0.0,None), softbounds=(0.0,5.0), precedence=(2),
        doc="""Ratio of width to height for the positive Gaussian pattern.""")

    negative_size = param.Number(default=0.25, bounds=(0.0,None), softbounds=(0.0,5.0), precedence=(3),
        doc="""Size of the negative Gaussian pattern.""")

    negative_aspect_ratio = param.Number(default=1.0, bounds=(0.0,None), softbounds=(0.0,5.0), precedence=(4),
        doc="""Ratio of width to height for the negative Gaussian pattern.""")

    sigmoid_slope = param.Number(default=10.0, bounds=(None,None), softbounds=(-100.0,100.0), precedence=(5),
        doc="""Parameter controlling the smoothness of the transition between the two regions;
            high values give a sharp transition.""")

    sigmoid_position = param.Number(default=0.0, bounds=(None,None), softbounds=(-1.0,1.0), precedence=(6),
        doc="""X position of the transition between the two regions.""")


    def function(self, p):
        diff_of_gaussians = DifferenceOfGaussians(positive_x=p.x, positive_y=p.y, negative_x=p.x, negative_y=p.y,
            positive_size=p.positive_size*p.size, positive_aspect_ratio=p.positive_aspect_ratio,
            negative_size=p.negative_size*p.size, negative_aspect_ratio=p.negative_aspect_ratio)

        sigmoid = Sigmoid(slope=p.sigmoid_slope, orientation=p.orientation+pi/2, x=p.x+p.sigmoid_position)

        return Composite(generators=[diff_of_gaussians, sigmoid], bounds=p.bounds,
            operator=np.multiply, xdensity=p.xdensity, ydensity=p.ydensity)()



class LogGaussian(PatternGenerator):
    """
    2D Log Gaussian pattern generator allowing standard gaussian
    patterns but with the added advantage of movable peaks.

    The spread governs decay rates from the peak of the Gaussian,
    mathematically this is the sigma term.

    The center governs the peak position of the Gaussian,
    mathematically this is the mean term.
    """

    aspect_ratio = param.Number(default=0.5, bounds=(0.0,None), inclusive_bounds=(True,False), softbounds=(0.0,1.0),
        doc="""Ratio of the pattern's width to height.""")

    x_shape = param.Number(default=0.8, bounds=(0.0,None), inclusive_bounds=(False,False), softbounds=(0.0,5.0),
        doc="""The length of the tail along the x axis.""")

    y_shape = param.Number(default=0.35, bounds=(0.0,None), inclusive_bounds=(False,False), softbounds=(0.0,5.0),
        doc="""The length of the tail along the y axis.""")


    def __call__(self, **params_to_override):
        """
        Call the subclass's 'function' method on a rotated and scaled
        coordinate system.

        Creates and fills an array with the requested pattern.  If
        called without any params, uses the values for the Parameters
        as currently set on the object. Otherwise, any params
        specified override those currently set on the object.
        """
        p = ParamOverrides(self, params_to_override)

        self._setup_xy(p)
        fn_result = self.function(p)
        self._apply_mask(p, fn_result)

        scale_factor = p.scale / np.max(fn_result)
        result = scale_factor*fn_result + p.offset

        for of in p.output_fns:
            of(result)

        return result


    def _setup_xy(self, p):
        """
        Produce pattern coordinate matrices from the bounds and
        density (or rows and cols), and transforms them according to
        x, y, and orientation.
        """
        self.debug("bounds=%s, xdensity=%s, ydensity=%s, x=%s, y=%s, orientation=%s",p.bounds, p.xdensity, p.ydensity, p.x, p.y, p.orientation)

        x_points,y_points = SheetCoordinateSystem(p.bounds, p.xdensity, p.ydensity).sheetcoordinates_of_matrixidx()

        self.pattern_x, self.pattern_y = self._create_and_rotate_coordinate_arrays(x_points-p.x, y_points-p.y, p)


    def _create_and_rotate_coordinate_arrays(self, x, y, p):
        """
        Create pattern matrices from x and y vectors, and rotate
        them to the specified orientation.
        """

        if p.aspect_ratio == 0 or p.size == 0:
            x = x * 0.0
            y = y * 0.0
        else:
            x = (x*10.0) / (p.size*p.aspect_ratio)
            y = (y*10.0) / p.size

        offset = np.exp(p.size)
        pattern_x = np.add.outer(np.sin(p.orientation)*y, np.cos(p.orientation)*x) + offset
        pattern_y = np.subtract.outer(np.cos(p.orientation)*y, np.sin(p.orientation)*x) + offset

        np.clip(pattern_x, 0, np.Infinity, out=pattern_x)
        np.clip(pattern_y, 0, np.Infinity, out=pattern_y)

        return pattern_x, pattern_y


    def function(self, p):
        return log_gaussian(self.pattern_x, self.pattern_y, p.x_shape, p.y_shape, p.size)



class SigmoidedDoLG(PatternGenerator):
    """
    Sigmoid multiplicatively combined with a difference of Log
    Gaussians, such that one part of the plane can be the mirror image
    of the other, and the peaks of the gaussians are movable.
    """

    size = param.Number(default=1.5)


    positive_size = param.Number(default=0.5, bounds=(0.0,None), inclusive_bounds=(True,False), softbounds=(0.0,10.0),
        doc="""Size of the positive LogGaussian pattern.""")

    positive_aspect_ratio = param.Number(default=0.5, bounds=(0.0,None), inclusive_bounds=(True,False), softbounds=(0.0,1.0),
        doc="""Ratio of width to height for the positive LogGaussian pattern.""")

    positive_x_shape = param.Number(default=0.8, bounds=(0.0,None), inclusive_bounds=(False,False), softbounds=(0.0,5.0),
        doc="""The length of the tail along the x axis for the positive LogGaussian pattern.""")

    positive_y_shape = param.Number(default=0.35, bounds=(0.0,None), inclusive_bounds=(False,False), softbounds=(0.0,5.0),
        doc="""The length of the tail along the y axis for the positive LogGaussian pattern.""")

    positive_scale = param.Number(default=1.5, bounds=(0.0,None), inclusive_bounds=(True,False), softbounds=(0.0,10.0),
        doc="""Multiplicative scale for the positive LogGaussian pattern.""")


    negative_size = param.Number(default=0.8, bounds=(0.0,None), inclusive_bounds=(True,False), softbounds=(0.0,10.0),
        doc="""Size of the negative LogGaussian pattern.""")

    negative_aspect_ratio = param.Number(default=0.3, bounds=(0.0,None), inclusive_bounds=(True,False), softbounds=(0.0,1.0),
        doc="""Ratio of width to height for the negative LogGaussian pattern.""")

    negative_x_shape = param.Number(default=0.8, bounds=(0.0,None), inclusive_bounds=(False,False), softbounds=(0.0,5.0),
        doc="""The length of the tail along the x axis for the negative LogGaussian pattern.""")

    negative_y_shape = param.Number(default=0.35, bounds=(0.0,None), inclusive_bounds=(False,False), softbounds=(0.0,5.0),
        doc="""The length of the tail along the y axis for the negative LogGaussian pattern.""")

    negative_scale = param.Number(default=1.0, bounds=(0.0,None), inclusive_bounds=(True,False), softbounds=(0.0,10.0),
        doc="""Multiplicative scale for the negative LogGaussian pattern.""")


    sigmoid_slope = param.Number(default=50.0, bounds=(None,None), softbounds=(-100.0,100.0),
        doc="""Parameter controlling the smoothness of the transition between the two regions;
            high values give a sharp transition.""")

    sigmoid_position = param.Number(default=0.05, bounds=(None,None), softbounds=(-1.0,1.0),
        doc="""X position of the transition between the two regions.""")


    def function(self, p):
        positive = LogGaussian(size=p.positive_size*p.size, aspect_ratio=p.positive_aspect_ratio, x_shape=p.positive_x_shape,
            y_shape=p.positive_y_shape, scale=p.positive_scale*p.scale, orientation=p.orientation, x=p.x, y=p.y,
            output_fns=[])

        negative = LogGaussian(size=p.negative_size*p.size, aspect_ratio=p.negative_aspect_ratio, x_shape=p.negative_x_shape,
            y_shape=p.negative_y_shape, scale=p.negative_scale*p.scale, orientation=p.orientation, x=p.x, y=p.y,
            output_fns=[])

        diff_of_log_gaussians = Composite(generators=[positive, negative], operator=np.subtract,
            xdensity=p.xdensity, ydensity=p.ydensity, bounds=p.bounds)

        sigmoid = Sigmoid(x=p.x+p.sigmoid_position, slope=p.sigmoid_slope, orientation=p.orientation+pi/2.0)

        return Composite(generators=[diff_of_log_gaussians, sigmoid], bounds=p.bounds,
            operator=np.multiply, xdensity=p.xdensity, ydensity=p.ydensity, output_fns=[DivisiveNormalizeL1()])()



class TimeSeries(param.Parameterized):
    """
    Generic class to return intervals of a discretized time series.
    """

    time_series = param.Array(default=np.repeat(np.array([0,1]),50),
        doc="""An array of numbers that form a series.""")

    sample_rate = param.Integer(default=50, allow_None=True, bounds=(0,None), inclusive_bounds=(False,False), softbounds=(0,44100),
        doc="""The number of samples taken per second to form the series.""")

    seconds_per_iteration = param.Number(default=0.1, bounds=(0.0,None), inclusive_bounds=(False,False), softbounds=(0.0,1.0),
        doc="""Number of seconds advanced along the time series on each iteration.""")

    interval_length = param.Number(default=0.1, bounds=(0.0,None), inclusive_bounds=(False,False), softbounds=(0.0,1.0),
        doc="""The length of time in seconds to be returned on each iteration.""")

    repeat = param.Boolean(default=True,
        doc="""Whether the signal loops or terminates once it reaches its end.""")


    def __init__(self, **params):
        super(TimeSeries, self).__init__(**params)
        self._next_interval_start = 0

        if self.seconds_per_iteration > self.interval_length:
            self.warning("Seconds per iteration > interval length, some signal will be skipped.")


    def append_signal(self, new_signal):
        self.time_series = np.hstack((self.time_series, new_signal))


    def extract_specific_interval(self, interval_start, interval_end):
        """
        Overload if special behaviour is required when a series ends.
        """

        interval_start = int(interval_start)
        interval_end = int(interval_end)

        if interval_start >= interval_end:
            raise ValueError("Requested interval's start point is past the requested end point.")

        elif interval_start > self.time_series.size:
            if self.repeat:
                interval_end = interval_end - interval_start
                interval_start = 0
            else:
                raise ValueError("Requested interval's start point is past the end of the time series.")

        if interval_end < self.time_series.size:
            interval = self.time_series[interval_start:interval_end]

        else:
            requested_interval_size = interval_end - interval_start
            remaining_signal = self.time_series[interval_start:self.time_series.size]

            if self.repeat:
                if requested_interval_size < self.time_series.size:
                    self._next_interval_start = requested_interval_size-remaining_signal.size
                    interval = np.hstack((remaining_signal, self.time_series[0:self._next_interval_start]))

                else:
                    repeated_signal = np.repeat(self.time_series, np.floor(requested_interval_size/self.time_series.size))
                    self._next_interval_start = requested_interval_size % self.time_series.size

                    interval = (np.hstack((remaining_signal, repeated_signal)))[0:requested_interval_size]

            else:
                self.warning("Returning last interval of the time series.")
                self._next_interval_start = self.time_series.size + 1

                samples_per_interval = self.interval_length*self.sample_rate
                interval = np.hstack((remaining_signal, np.zeros(samples_per_interval-remaining_signal.size)))

        return interval


    def __call__(self):
        interval_start = self._next_interval_start
        interval_end = int(np.floor(interval_start + self.interval_length*self.sample_rate))

        self._next_interval_start += int(np.floor(self.seconds_per_iteration*self.sample_rate))
        return self.extract_specific_interval(interval_start, interval_end)



def generate_sine_wave(duration, frequency, sample_rate):
    time_axis = np.linspace(0.0, duration, int(duration*sample_rate))
    return np.sin(2.0*pi*frequency * time_axis)



class TimeSeriesParam(ClassSelector):
    """
    Parameter whose value is a TimeSeries object.
    """

    def __init__(self, **params):
        super(TimeSeriesParam, self).__init__(TimeSeries, **params)



class PowerSpectrum(PatternGenerator):
    """
    Outputs the spectral density of a rolling interval of the input
    signal each time it is called. Over time, the results could be
    arranged into a spectrogram, e.g. for an audio signal.
    """

    x = param.Number(precedence=(-1))
    y = param.Number(precedence=(-1))
    size = param.Number(precedence=(-1))
    orientation = param.Number(precedence=(-1))

    scale = param.Number(default=0.01, bounds=(0,None), inclusive_bounds=(False,False), softbounds=(0.001,1000),
        doc="""The amount by which to scale amplitudes by. This is useful if we want to rescale to say a range [0:1].

        Note: Constant scaling is preferable to dynamic scaling so as not to artificially ramp down loud sounds while ramping
        up hiss and other background interference.""")

    signal = TimeSeriesParam(default=TimeSeries(time_series=generate_sine_wave(0.1,5000,20000), sample_rate=20000),
        doc="""A TimeSeries object on which to perfom the Fourier Transform.""")

    min_frequency = param.Integer(default=0, bounds=(0,None), inclusive_bounds=(True,False), softbounds=(0,10000),
        doc="""Smallest frequency for which to return an amplitude.""")

    max_frequency = param.Integer(default=9999, bounds=(0,None), inclusive_bounds=(False,False), softbounds=(0,10000),
        doc="""Largest frequency for which to return an amplitude.""")

    windowing_function = param.Parameter(default=None,
        doc="""This function is multiplied with the current interval, i.e. the most recent portion of the
        waveform interval of a signal, before performing the Fourier transform.  It thus shapes the interval,
        which is otherwise always rectangular.

        The function chosen here dictates the tradeoff between resolving comparable signal strengths with similar
        frequencies, and resolving disparate signal strengths with dissimilar frequencies.

        numpy provides a number of options, e.g. bartlett, blackman, hamming, hanning, kaiser; see
        http://docs.scipy.org/doc/numpy/reference/routines.window.html

        You may also supply your own.""")


    def __init__(self, **params):
        super(PowerSpectrum, self).__init__(**params)

        self._previous_min_frequency = self.min_frequency
        self._previous_max_frequency = self.max_frequency


    def _create_frequency_indices(self):
        if self.min_frequency >= self.max_frequency:
            raise ValueError("PowerSpectrum: min frequency must be lower than max frequency.")

        # calculate the discrete frequencies possible for the given sample rate.
        sample_rate = self.signal.sample_rate
        available_frequency_range = np.fft.fftfreq(sample_rate, d=1.0/sample_rate)[0:sample_rate/2]

        if not available_frequency_range.min() <= self.min_frequency or not available_frequency_range.max() >= self.max_frequency:
            raise ValueError("Specified frequency interval [%s:%s] is unavailable, available range is [%s:%s]. Adjust to these frequencies or modify the sample rate of the TimeSeries object." %(self.min_frequency, self.max_frequency, available_frequency_range.min(), available_frequency_range.max()))

        min_freq = np.nonzero(available_frequency_range >= self.min_frequency)[0][0]
        max_freq = np.nonzero(available_frequency_range <= self.max_frequency)[0][-1]

        self._set_frequency_spacing(min_freq, max_freq)


    def _set_frequency_spacing(self, min_freq, max_freq):
        """
        Frequency spacing to use, i.e. how to map the available
        frequency range to the discrete sheet rows.

        NOTE: We're calculating the spacing of a range between the
        highest and lowest frequencies, the actual segmentation and
        averaging of the frequencies to fit this spacing occurs in
        _getAmplitudes().

        This method is here solely to provide a minimal overload if
        custom spacing is required.
        """

        self.frequency_spacing = np.linspace(min_freq, max_freq, num=self._sheet_dimensions[0]+1, endpoint=True)


    def _get_row_amplitudes(self):
        """
        Perform a real Discrete Fourier Transform (DFT; implemented
        using a Fast Fourier Transform algorithm, FFT) of the current
        sample from the signal multiplied by the smoothing window.

        See numpy.rfft for information about the Fourier transform.
        """

        signal_interval = self.signal()
        sample_rate = self.signal.sample_rate

        # A signal window *must* span one sample rate
        signal_window = np.tile(signal_interval, int(np.ceil(1.0/self.signal.interval_length)))

        if self.windowing_function:
            smoothed_window = signal_window[0:sample_rate] * self.windowing_function(sample_rate)
        else:
            smoothed_window = signal_window[0:sample_rate]

        amplitudes = (np.abs(np.fft.rfft(smoothed_window))[0:sample_rate/2] + self.offset) * self.scale

        for index in range(0, self._sheet_dimensions[0]-2):
            start_frequency = self.frequency_spacing[index]
            end_frequency = self.frequency_spacing[index+1]

            normalisation_factor =  end_frequency - start_frequency
            if normalisation_factor == 0:
                amplitudes[index] = amplitudes[start_frequency]
            else:
                amplitudes[index] = np.sum(amplitudes[int(start_frequency):int(end_frequency)]) / normalisation_factor

        return np.flipud(amplitudes[0:self._sheet_dimensions[0]].reshape(-1,1))


    def set_matrix_dimensions(self, bounds, xdensity, ydensity):
        super(PowerSpectrum, self).set_matrix_dimensions(bounds, xdensity, ydensity)

        self._sheet_dimensions = SheetCoordinateSystem(bounds, xdensity, ydensity).shape
        self._create_frequency_indices()


    def _shape_response(self, row_amplitudes):
        if self._sheet_dimensions[1] > 1:
            row_amplitudes = np.repeat(row_amplitudes, self._sheet_dimensions[1], axis=1)

        return row_amplitudes


    def __call__(self):
        if self._previous_min_frequency != self.min_frequency or self._previous_max_frequency != self.max_frequency:
            self._previous_min_frequency = self.min_frequency
            self._previous_max_frequency = self.max_frequency
            self._create_frequency_indices()

        return self._shape_response(self._get_row_amplitudes())



class Spectrogram(PowerSpectrum):
    """
    Extends PowerSpectrum to provide a temporal buffer, yielding a 2D
    representation of a fixed-width spectrogram.
    """

    min_latency = param.Integer(default=0, precedence=1,
        bounds=(0,None), inclusive_bounds=(True,False), softbounds=(0,1000),
        doc="""Smallest latency (in milliseconds) for which to return amplitudes.""")

    max_latency = param.Integer(default=500, precedence=2,
        bounds=(0,None), inclusive_bounds=(False,False), softbounds=(0,1000),
        doc="""Largest latency (in milliseconds) for which to return amplitudes.""")


    def __init__(self, **params):
        super(Spectrogram, self).__init__(**params)

        self._previous_min_latency = self.min_latency
        self._previous_max_latency = self.max_latency


    def _shape_response(self, new_column):

        millisecs_per_iteration = int(self.signal.seconds_per_iteration * 1000)

        if millisecs_per_iteration > self.max_latency:
            self._spectrogram[0:,0:] = new_column
        else:
            # Slide old values along, add new data to left hand side.
            self._spectrogram[0:, millisecs_per_iteration:] = self._spectrogram[0:, 0:self._spectrogram.shape[1]-millisecs_per_iteration]
            self._spectrogram[0:, 0:millisecs_per_iteration] = new_column

        sheet_representation = np.zeros(self._sheet_dimensions)

        for column in range(0,self._sheet_dimensions[1]):
            start_latency = int(self._latency_spacing[column])
            end_latency = int(self._latency_spacing[column+1])

            normalisation_factor = end_latency - start_latency
            if normalisation_factor > 1:
                sheet_representation[0:, column] = np.sum(self._spectrogram[0:, start_latency:end_latency], axis=1) / normalisation_factor
            else:
                sheet_representation[0:, column] = self._spectrogram[0:, start_latency]

        return sheet_representation


    def set_matrix_dimensions(self, bounds, xdensity, ydensity):
        super(Spectrogram, self).set_matrix_dimensions(bounds, xdensity, ydensity)
        self._create_latency_indices()


    def _create_latency_indices(self):
        if self.min_latency >= self.max_latency:
            raise ValueError("Spectrogram: min latency must be lower than max latency.")

        self._latency_spacing = np.floor(np.linspace(self.min_latency, self.max_latency, num=self._sheet_dimensions[1]+1, endpoint=True))
        self._spectrogram = np.zeros([self._sheet_dimensions[0],self.max_latency])


    def __call__(self):
        if self._previous_min_latency != self.min_latency or self._previous_max_latency != self.max_latency:
            self._previous_min_latency = self.min_latency
            self._previous_max_latency = self.max_latency
            self._create_latency_indices()

        return super(Spectrogram, self).__call__()




_public = list(set([_k for _k,_v in locals().items() if isinstance(_v,type) and issubclass(_v,PatternGenerator)]))
__all__ = _public + ["image", "random", "patterncoordinator", "boundingregion", "sheetcoords"]
# Should avoid loading audio.py and other modules that rely on external
# libraries that might not be present on this system.
