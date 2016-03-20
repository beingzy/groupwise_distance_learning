""" objective introduction
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/MM/DD
"""
import unittest
import numpy as np
from numpy.random import normal
from groupwise_distance_learning.kstest import kstest_2samp_greater


class TestKSTest2SampGreater(unittest.TestCase):

    def setUp(self):
        # ensure reproducibility by setting random seed for random.method
        np.random.seed(1234)
        nsize = 100
        self._x = normal(0, 1, nsize)
        self._y = normal(0.5, 1, nsize)
        self._z = normal(0.1, 2, nsize)

    def test_xgreatery(self):
        # accept null hypothesis
        res_ts, res_pval = kstest_2samp_greater(self._x, self._y)

        print("res_ts: {}".format(res_ts))
        print("res_pval: {}".format(res_pval))

        if res_pval > 0.05:
            print("SUCCCESS: null hypothesis (x is not less than y) should hold (alpha=0.05)!")
            is_ok = True
        else:
            is_ok = False

        self.assertTrue(is_ok)

    def test_ygreaterx(self):
        # reject null hypothesis
        res_ts, res_pval = kstest_2samp_greater(self._y, self._x)
        true_ts = 0.19
        true_pval = 0.03
        is_match = round(res_ts, 2) == true_ts and round(res_pval, 2) == true_pval

        print("res_tes: {}".format(res_ts))
        print("res_tes: {}".format(res_pval))

        if res_pval < 0.05:
            print("SUCCCESS: null hypothesis (y > x) should be rejected (alpha=0.05)!")

        self.assertTrue(is_match)

    def test_xgreaterz(self):
        # accept null hypothesis
        res_ts, res_pval = kstest_2samp_greater(self._x, self._z)
        true_ts = 0.11
        true_pval = 0.3
        is_match = round(res_ts, 2) == true_ts and round(res_pval, 2) == true_pval

        print("res_tes: {}".format(res_ts))
        print("res_tes: {}".format(res_pval))

        if res_pval > 0.05:
            print("SUCCCESS: null hypothesis (x > z) should be not rejected (alpha=0.05)!")

        self.assertTrue(is_match)


if __name__ == "__main__":
    unittest.main()