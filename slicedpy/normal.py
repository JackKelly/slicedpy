from __future__ import division, print_function
import math
import scipy.stats as stats
import numpy as np
from scipy.stats import norm
from slicedpy.bunch import Bunch

class Normal(Bunch):
    """
    Attributes:
      (see reset())
    """

    def __init__(self, values=None, **kwds):
        """
        Args
          * values (np.ndarray) (optional)
        """
        super(Normal, self).__init__(**kwds)
        self.reset()
        if values is not None:
            self.partial_fit(values)

    def reset(self):
        self._fit_has_been_called = False
        self.min = None
        self.max = None
        self.mean = None
        self.var = None
        self.size = 0
        self._M2 = 0
        return self

    @property
    def std(self):
        return math.sqrt(self.var)

    def fit(self, values):
        _sanity_check_values(values)
        self._fit_has_been_called = True
        self.size = values.size
        self.var = values.var(ddof=1)
        self.mean = values.mean()
        self.min = values.min()
        self.max = values.max()
        return self

    def partial_fit(self, values):
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        if self._fit_has_been_called:
            raise Exception('You cannot call partial_fit() if fit() has been'
                            ' called on this object!')

        _sanity_check_values(values)

        if self.mean is None:
            self.mean = 0

        for x in values:
            self.size += 1
            delta = x - self.mean
            self.mean += delta / self.size
            self._M2 += delta * (x - self.mean)

        self.var = self._M2 / (self.size - 1)

        if self.min is None:
            self.min = values.min()
            self.max = values.max()
        else:
            self.min = min(values.min(), self.min)
            self.max = max(values.max(), self.max)

        return self

    def score(self, x):
        return norm.logpdf(x, loc=self.mean, scale=self.std)

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
        """Returns True is self and other have similar means."""
        return (math.fabs(self.mean - other.mean) < mean_delta_threshold or
                self.welch_ttest(other) > p_value_threshold)

    def rough_combination(self, other):
        """Note that this gives the correct answer for ``size`` and ``mean``
        but will give the incorrect answer for ``var`` if the variables are
        correlated!!

        Does not alter self.  Instead returns new instance of :class:`Normal`.
        """
        new = Normal()
        new.size = self.size + other.size
        new.mean = ((self.mean * self.size) + (other.mean * other.size)) / new.size
        # http://www.emathzone.com/tutorials/basic-statistics/combined-variance.html
        new.var = (((self.size * (self.var + (self.mean - new.mean)**2)) +
                    (other.size * (other.var + (other.mean - new.mean)**2))) / (new.size+2))
        new.min = min(self.min, other.min)
        new.max = max(self.max, other.max)
        return new


def _sanity_check_values(values):
    if not isinstance(values, np.ndarray):
        raise TypeError('values must be an np.ndarray')
