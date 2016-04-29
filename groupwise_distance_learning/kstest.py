""" KStest function depending on R-core
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/29
"""
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

from numpy import array, asarray, exp, min
from numpy.random import choice
import numpy as np

# set up environment
robjects.conversion.py2ri = numpy2ri
rstats = importr("stats")


def kstest_2samp_greater(x, y, auto_adjust=False):
    """ return tests staticics and p-value for KS-tests
    which compare two samples.
    IF either x or y has too few elements, the test will
    return None as test statistics, 1 as pvalue.

    Hypothesis:
    H0: distribution of X >= distribution of Y
    H1: distribution of X < distribution of Y

    Parameters:
    ----------
    * x: <vector-like, numeric> samples from one population
    * y: <vector-like, numeric> samples from one population
    * auto_adjust: <boolean> True: sample y to create more balance
         size

    Returns:
    -------
    * ts: <numeric> tests statistics of KS-tests
    * pvalue: <numeric> p-value fo tests statistics
    """
    if isinstance(x, np.ndarray):
        x = x.ravel()
    else:
        x = np.array(x)

    if isinstance(y, np.ndarray):
        y = y.ravel()
    else:
        y = np.array(y)

    try:
        x_size = len(x)
        y_size = len(y)
        if (x_size > y_size) and auto_adjust:
            x = choice(x, size=y_size, replace=False)
        if (y_size > x_size) and auto_adjust:
            y = choice(x, size=x_size, replace=False)

        setting = array(["less"], dtype="str")
        test_res = rstats.ks_test(x, y, alternative=setting)
        ts, pval = test_res[0][0], test_res[1][0]
    except:
        ts, pval = None, 0

    return ts, pval