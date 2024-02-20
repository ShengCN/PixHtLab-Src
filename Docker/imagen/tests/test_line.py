from numpy import pi, where

import imagen
from holoviews.core.boundingregion import BoundingBox

try:
    import unittest2 as unittest
except ImportError:
    import unittest

class LineTest(unittest.TestCase):
    """Tests the behavior of the line pattern,
    particularly for small thicknesses"""

    def minimal_line_thickness(self,c):
        """
        help function used by the test_* functions.

        It checks that the line width is of one pixel
        """
        radius = c['radius']
        bounds=BoundingBox(radius = radius)
        xdensity = c['density']
        ydensity = c['density']
        x=c['x']
        for orientation in [0, pi/4, pi/2]:
            c['orientation'] = orientation
            im = imagen.Line(
                scale=1,offset=0,orientation=orientation,
                enforce_minimal_thickness=True,
                thickness=0.0,x=x, y=0.,
                smoothing=0.0,
                xdensity=xdensity, ydensity=ydensity,
                bounds=bounds)()
            number_of_ones=(im==1).sum()
            if orientation==pi/4:
                y_crossing=where(im[:,-1]>=1)[0][0]
            else:
                y_crossing=0.0
            width_in_pixels = 2*radius*ydensity-y_crossing
            msg = """
                Thickness of the line should be of one pixel.
                Actual line has an everage thickness of: %f.
                Experimental conditions:
                """ % (number_of_ones/float(width_in_pixels)) + str(c)
            self.assertEqual(
                number_of_ones, width_in_pixels,
                msg=msg
                )

    def test_minimal_line_thickness_density20_x_jitter(self):
        c = {'radius':5,  'density': 20,  'x': -0.001/20}; self.minimal_line_thickness(c)

    def test_minimal_line_thickness_density20_x0(self):
        c = {'radius':5,  'density': 20,  'x': 0.}; self.minimal_line_thickness(c)

    def test_minimal_line_thickness_density100_x0(self):
        c = {'radius':1,  'density': 100, 'x': 0.}; self.minimal_line_thickness(c)

    def test_minimal_line_thickness_density1000_x0(self):
        c = {'radius':0.1,'density': 1000,'x': 0.}; self.minimal_line_thickness(c)

    def test_minimal_line_thickness_density20_x2(self):
        c = {'radius':5,  'density': 20,  'x': 2.}; self.minimal_line_thickness(c)

if __name__ == '__main__':
    unittest.main()
