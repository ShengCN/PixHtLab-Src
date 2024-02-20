"""
Unit tests for time-dependent random imagen patterns.
"""

from imagen.random import UniformRandom, RandomGenerator
import unittest
import param

import numpy as np
from numpy.testing import assert_almost_equal


class TestTimeDependentRandom(unittest.TestCase):
    """
    These tests focus on the UniformRandom pattern as an example of
    time-dependent random streams.
    """

    def setUp(self):
        param.Dynamic.time_dependent=None
        self.time_fn= param.Time(time_type=int)


        self.non_td_t0 = np.array([[ 0.42126358,  0.31223736,  0.6377056 ],
                                   [ 0.39064891,  0.41890129,  0.03101435],
                                   [ 0.67184411,  0.09738222,  0.46043459]])

        self.non_td_t1 = np.array([[ 0.26362267,  0.55429345,  0.46786591],
                                   [ 0.10548665,  0.35680933,  0.95669633],
                                   [ 0.9689579 ,  0.77990637,  0.55152049]])

        self.td_p1_t0 = np.array([[ 0.33532388,  0.60655109,  0.64335319],
                                  [ 0.79866631,  0.97845841,  0.81552961],
                                  [ 0.80266319,  0.22363254,  0.00939643]])

        self.td_p2_t0 = np.array([[ 0.95266048,  0.13381049,  0.69471505],
                                  [ 0.12987513,  0.67797684,  0.00453637],
                                  [ 0.09066964,  0.31480213,  0.67968573]])

        self.td_p1_t1 = np.array([[ 0.60069123,  0.28276134,  0.21280783],
                                  [ 0.57636133,  0.53144412,  0.29766811],
                                  [ 0.60853213,  0.06351663,  0.03819535]])

        self.td_p2_t1 = np.array([[ 0.40411503,  0.26346125,  0.39546975],
                                  [ 0.95572014,  0.28943148,  0.26373155],
                                  [ 0.61059597,  0.81626189,  0.16745686]])

    def test_non_time_dependent(self):
        param.Dynamic.time_dependent=None
        param.Dynamic.time_fn = self.time_fn
        pattern = UniformRandom(name='test1', xdensity=3, ydensity=3)

        t0 = pattern()
        t1 = pattern()

        assert_almost_equal(t0, self.non_td_t0, err_msg="UniformRandom output doesn't match reference (t0).")
        if np.allclose(t0, t1):
            raise self.failureException("UniformRandom output hasn't changed between calls.")
        assert_almost_equal(t1, self.non_td_t1, err_msg="UniformRandom output doesn't match reference (t1).")


    def test_time_dependent_t0(self):
        RandomGenerator.time_dependent = True
        RandomGenerator.time_fn = self.time_fn

        self.assertEqual(self.time_fn(), 0)

        pattern1 = UniformRandom(name='test1', xdensity=3, ydensity=3)
        pattern2 = UniformRandom(name='test2', xdensity=3, ydensity=3)

        p1_t0 = pattern1()
        p2_t0 = pattern2()

        assert_almost_equal(p1_t0, self.td_p1_t0,
                            err_msg="UniformRandom output doesn't match reference (pattern 1 @ t0).")

        assert_almost_equal(p2_t0, self.td_p2_t0,
                            err_msg="UniformRandom output doesn't match reference  (pattern 2 @ t0).")

        assert_almost_equal(pattern1(), p1_t0,
                            err_msg = "Output shouldn't change between calls (pattern 1).")

        assert_almost_equal(pattern2(), p2_t0,
                            err_msg = "Output shouldn't change between calls (pattern 2).")

        if np.allclose(pattern1(), pattern2()):
            raise self.failureException("UniformRandom with different names should have different output")


    def test_time_dependent_t1(self):
        RandomGenerator.time_dependent = True
        RandomGenerator.time_fn = self.time_fn

        self.time_fn.advance(1)
        self.assertEqual(self.time_fn(), 1)

        pattern1 = UniformRandom(name='test1', xdensity=3, ydensity=3)
        pattern2 = UniformRandom(name='test2', xdensity=3, ydensity=3)

        p1_t1 = pattern1()
        p2_t1 = pattern2()

        assert_almost_equal(p1_t1, self.td_p1_t1,
                            err_msg="UniformRandom output doesn't match reference (pattern 1 @ t1).")

        assert_almost_equal(p2_t1, self.td_p2_t1,
                            err_msg="UniformRandom output doesn't match reference  (pattern 2 @ t1).")

        assert_almost_equal(pattern1(), p1_t1,
                            err_msg = "Output shouldn't change between calls (pattern 1).")

        assert_almost_equal(pattern2(), p2_t1,
                            err_msg = "Output shouldn't change between calls (pattern 2).")

        if np.allclose(pattern1(), pattern2()):
            raise self.failureException("UniformRandom with different names should have different output")


    def test_time_dependent_time_switch_pattern1(self):
        RandomGenerator.time_dependent = True
        RandomGenerator.time_fn = self.time_fn

        pattern1 = UniformRandom(name='test1', xdensity=3, ydensity=3)

        p1_t0 = pattern1()
        self.time_fn.advance(1)
        p1_t1 = pattern1()
        self.time_fn.advance(-1)
        p1_t0_2 = pattern1()

        assert_almost_equal(p1_t0, self.td_p1_t0,
                            err_msg="UniformRandom output doesn't match reference (pattern 1 @ t0).")

        assert_almost_equal(p1_t1, self.td_p1_t1,
                            err_msg="UniformRandom output doesn't match reference (pattern 1 @ t1).")

        assert_almost_equal(p1_t0_2, self.td_p1_t0,
                            err_msg="UniformRandom output doesn't match reference (pattern 1 @ t0).")



if __name__ == "__main__":
    import nose
    nose.runmodule()
