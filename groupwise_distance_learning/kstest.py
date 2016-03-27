""" KStest function depending on R-core
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/29
"""
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

from numpy import array, asarray, exp, min
from numpy.random import choice
# import numpy as np
# import pandas as pd

# set up environment
robjects.conversion.py2ri = numpy2ri
rstats = importr("stats")

def kstest_2samp_greater(x, y):
    """ return tests staticics and p-value for KS-tests
    which compare two samples.

    Hypothesis:
    H0: distribution of X >= distribution of Y
    H1: distribution of X < distribution of Y

    Parameters:
    ----------
    * x: <vector-like, numeric> samples from one population
    * y: <vector-like, numeric> samples from one population

    Returns:
    -------
    * ts: <numeric> tests statistics of KS-tests
    * pvalue: <numeric> p-value fo tests statistics
    """
    # sampling y
    # n, m = y.shape
    # msg = "y's type; {}, nrow: {}, ncol: {}".format(type(y), n, m)
    # print(msg)

    # if len(y) > 1.2 * len(x):
    #     y = choice(y, size=len(x), replace=False)

    setting = array(["less"], dtype="str")
    test_res = rstats.ks_test(x, y, alternative=setting)
    ts, pval = test_res[0][0], test_res[1][0]
    return ts, pval

# def kstest_2samp_greater(data1, data2):
#     """KS-test with two-sample, null-hypothesis: X > Y"""
#     # data1, data2 = map(asarray, (data1, data2))
#
#     print("data1's type: ".format(type(data1)))
#     print("data1's type: ".format(type(data1)))
#
#     if isinstance(data1, pd.Series):
#         data1 = data1.tolist()
#     if isinstance(data2, pd.Series):
#         data2 = data2.tolist()
#
#     n1 = len(data1)
#     n2 = len(data2)
#     n = n1 * n2 / float(n1 + n2)
#     data_all = np.concatenate([data1, data2])
#     # return order index
#     order_idx = data_all.argsort(axis=0)
#     z = np.cumsum([1.0 / n1 if (idx <= n1) else -1.0 / n2 for idx in order_idx])
#
#     if len(np.unique(data_all)) < (n1 + n2):
#         diffs = np.diff(z)
#         idx = [ii for ii, diff in zip(range(len(diffs)), diffs) if diff]
#         idx.append(n1 + n2 - 1)
#         z = z[idx]
#
#     test_stat = abs(-np.min(z))
#     pval = min([1, max([0, exp(-2 * n * test_stat * test_stat)])])
#     return test_stat, pval