from __future__ import division, print_function
import math
import scipy.stats as stats
import numpy as np
from slicedpy.bunch import Bunch

class Normal(Bunch):
    """
    Attributes:
      * min
      * max
      * mean
      * var (variance)
      * size
    """

    def __init__(self, values=None, **kwds):
        """
        Args
          * values (np.ndarray) (optional)
        """
        super(Normal, self).__init__(**kwds)
        if values is not None:
            if not isinstance(values, np.ndarray):
                raise TypeError('values must be an np.ndarray')
            else:
                self.size = values.size
                self.var = values.var(ddof=1)
                self.mean = values.mean()
                self.min = values.min()
                self.max = values.max()

    def welch_ttest(self, other):

        # http://en.wikipedia.org/wiki/Welch%27s_t_test
        # http://stackoverflow.com/questions/10038543/tracking-down-the-assumptions-made-by-scipys-ttest-ind-function

        mean1 = self.mean
        var1 = self.var
        n1 = self.size
        mean2 = other.mean
        var2 = other.var
        n2 = other.size

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


    def similar_mean(self, other, p_value_threshold=0.1, mean_delta_threshold=5):
        return (math.fabs(self.mean - other.mean) < mean_delta_threshold or
                self.welch_ttest(other) > p_value_threshold)


    def rough_mean_of_two_normals(self, other):
        """Note that this gives the correct answer for ``size`` and ``mean``
        but will give the incorrect answer for ``var`` if the variables are
        correlated!!

        Does not alter self.  Instead returns new :class:`Normal`.
        """
        new_size = self.size + other.size
        new_mean = ((self.mean * self.size) + (other.mean * other.size)) / new_size
        # http://www.emathzone.com/tutorials/basic-statistics/combined-variance.html
        new_var = (((self.size * (self.var + (self.mean - new_mean)**2)) +
                    (other.size * (other.var + (other.mean - new_mean)**2))) / (new_size+2))
        return Normal(size=new_size, mean=new_mean, var=new_var)
