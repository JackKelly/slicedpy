from __future__ import division, print_function
import math
import numpy as np
import scipy.stats as stats
from slicedpy.bunch import Bunch

def welch_ttest(b1, b2):
    """
    Args:
       b1, b2 are two Bunches with the following fields:
       * mean: sample mean
       * var: sample variance
       * size: sample size

    http://en.wikipedia.org/wiki/Welch%27s_t_test
    http://stackoverflow.com/questions/10038543/tracking-down-the-assumptions-made-by-scipys-ttest-ind-function

    Test with:
    rvs1 = stats.norm.rvs(loc=5,scale=10,size=5000)
    rvs2 = stats.norm.rvs(loc=5,scale=10,size=5000)
    b1 = Bunch(mean=rvs1.mean(), var=rvs1.var(ddof=1), size=rvs1.size)
    b1 = Bunch(mean=rvs2.mean(), var=rvs2.var(ddof=1), size=rvs2.size)
    welch_ttest(b1, b1)
    # Should be almost equal to:
    stats.ttest_ind(rvs1, rvs2, equal_var=False)[1]
    """

    mean1 = b1.mean
    var1 = b1.var
    n1 = b1.size
    mean2 = b2.mean
    var2 = b2.var
    n2 = b2.size

    # Calculate t statistic
    var_div_n = (var1 / n1) + (var2 / n2)
    t_stat = (mean1 - mean2) / math.sqrt(var_div_n)
    
    # Welch-Satterthwaite degrees of freedom:
    v1 = n1 - 1
    v2 = n2 - 1
    df = var_div_n**2 / ((var1**2.0 / (n1**2 * v1)) + 
                         (var2**2.0 / (n2**2 * v2)))

    # T-test:
    two_tailed_p_value = 1.0 - (stats.t.cdf( math.fabs(t_stat), df) - 
                                stats.t.cdf(-math.fabs(t_stat), df))
    return two_tailed_p_value


def similar_mean(b1, b2, p_value_threshold=0.1, mean_delta_threshold=5):
    """
    Args
      * b1, b2: Bunches describing normal distributions, each with fields 
        ``mean``, ``var``, ``size``
    """
    return (math.fabs(b1.mean - b2.mean) < mean_delta_threshold or
            welch_ttest(b1, b2) > p_value_threshold)


def rough_mean_of_two_normals(b1, b2):
    """Note that this gives the correct answer for ``size`` and ``mean``
    but will give the incorrect answer for ``var`` if the variables are
    correlated!!
    """
    new_size = b1.size + b2.size
    new_mean = ((b1.mean * b1.size) + (b2.mean * b2.size)) / new_size
    # http://www.emathzone.com/tutorials/basic-statistics/combined-variance.html
    new_var = (((b1.size * (b1.var + (b1.mean - new_mean)**2)) +
                (b2.size * (b2.var + (b2.mean - new_mean)**2))) / (new_size+2))
    return Bunch(size=new_size, mean=new_mean, var=new_var)
