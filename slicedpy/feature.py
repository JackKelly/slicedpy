from __future__ import print_function, division
import matplotlib.dates as mdates
import scipy.stats as stats
from normal import Normal

class Feature(Normal):
    """All feature detectors output a list of :class:`Feature` objects.
    The idea is that all Features must have a ``start`` and an ``end``
    timestamp plus zero or more other parameters specific to that
    feature detector.

    Attributes:
        * ``start`` (int): index into data array holding power data
        * ``end`` (int): index into data array holding power data
        * ``_p_value_both_halves`` (float): set by ttest_both_halves()

    """
    def __init__(self, start, end, **kwds):
        self.start = start
        self.end = end
        self.size = self.end - self.start
        super(Feature, self).__init__(**kwds)

    def ttest_both_halves(self, data):
        """
        Test if the left and right half of the data masked by this 
        Feature have the same mean or not.

        Returns and stores a two-tailed p-value.  
        Stored in ``self.p_value_both_halves``

        Args:
          * data (np.ndarray)
        """
        width = self.end - self.start
        half_way = int((width / 2) + self.start)
        left = data[self.start:half_way]
        right = data[half_way:self.end]
        self.p_value_both_halves = stats.ttest_ind(left, right)[1]            
        return self.p_value_both_halves

    def linregress(self, series):
        """
        Linear regression of data masked by this feature.

        Returns nothing.  Instead sets member variables:
          * ``self.slope`` (in units of watts per second)
          * ``self.r_value``
          * ``self.p_value``
          * ``self.stderr``

        Args
          * series (:class:`pandas.Series`)
        """
        ss = series[self.start:self.end]
        x = mdates.date2num(ss.index) * mdates.SEC_PER_DAY
        (self.slope, _, self.r_value, 
         self.p_value, self.stderr) = stats.linregress(x, ss.values)
