""" objective introduction
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/MM/DD
"""
import unittest
from numpy.random import normal
from groupwise_distance_learning.kstest import kstest_2samp_greater


class TestKSTest2SampGreater(unittest.TestCase):

    def setUp(self):
        nsize = 100
        self._x = normal(0, 1, nsize)
        self._y = normal(0.5, 1, nsize)
        self._z = normal(0.1, 2, nsize)

    def test_xgreatery(self):
        # accept null hypothesis
        res_ts, res_pval = kstest_2samp_greater(self._x, self._y)
        true_ts = None
        true_pval = None
        is_match = res_ts == true_ts and res_pval and true_pval
        self.assertTrue(False)

    def test_ygreaterx(self):
        # reject null hypothesis
        res_ts, res_pval = kstest_2samp_greater(self._y, self._x)
        true_ts = None
        true_pval = None
        is_match = res_ts == true_ts and res_pval and true_pval
        self.assertTrue(False)

    def test_xgreaterz(self):
        # accept null hypothesis
        res_ts, res_pval = kstest_2samp_greater(self._x, self._z)
        true_ts = None
        true_pval = None
        is_match = res_ts == true_ts and res_pval and true_pval
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()