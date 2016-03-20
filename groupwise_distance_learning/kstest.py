""" KStest function depending on R-core
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/29
"""
from numpy import array
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

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
    setting = array(["less"], dtype="str")
    test_res = rstats.ks_test(x, y, alternative=setting)
    ts, pval = test_res[0][0], test_res[1][0]
    return ts, pval