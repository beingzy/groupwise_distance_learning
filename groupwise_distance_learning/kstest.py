""" KStest function depending on R-core
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/29
"""
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

# set up environment
robjects.conversion.py2ri = numpy2ri
rstats = importr("stats")

def kstest_2samp_greater(x, y):
    """ return test staticics and p-value for KS-test
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
    * ts: <numeric> test statistics of KS-test
    * pvalue: <numeric> p-value fo test statistics
    """
    setting = np.array(["less"], dtype="str")
    test_res = rstats.ks_test(x, y, alternative=setting)
    ts, pval = test_res[0][0], test_res[1][0]
    return ts, pval