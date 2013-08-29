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

