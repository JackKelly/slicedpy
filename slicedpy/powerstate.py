from __future__ import print_function, division
from bunch import Bunch
import copy
from slicedpy.normal import Normal
from slicedpy.datastore import DataStore
import matplotlib.dates as mdates
import numpy as np

class PowerState(Bunch):
    """
    A washing machine might have three power states: washing, heating,
    spinning.

    Attributes:
        * start: datetime of start of each power state
        * end: datetime of end of each power state
        * duration: DataStore (GMM)
        * power: DataStore (Normal)
        * slope: DataStore (GMM)
        * intercept (float)
        * spike_histogram: 2D DataStore (GMM), one col per bin 
          (don't bother recording bin edges, assume these remain constant
           in fact, put bin edges in a config.py file)
        * count_per_run = DataStore (GMM): number of times this power state is seen per run 

    """
    def __init__(self,  **kwds):
        super(PowerState, self).__init__(**kwds)

    def similar(self, other):
        return self.power.model.similar_mean(other.power.model)

    def merge(self, other):
        """Merges ``other`` with ``self``."""
        pass

    def plot(self, ax, color='k'):
        ax.plot([self.start, self.end], 
                [self.power.model.mean, self.power.model.mean], color=color)
        
        if self.__dict__.get('slope') is not None:
            print("plotting slope: intercept=", self.intercept, 
                  "slope=", self.slope)
            curve = lambda x, c, m: c + (m / x)
            num_start = mdates.date2num(self.start)
            num_end = num_start + (10 / mdates.SEC_PER_DAY)
            X = np.linspace(num_start, num_end, 10)
            x = X * mdates.SEC_PER_DAY
            ax.plot(X, 
                    curve((x-x[0])+1, self.intercept, self.slope),
                    color=color)
