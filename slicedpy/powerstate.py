from __future__ import print_function, division
from bunch import Bunch
import copy
from slicedpy.normal import Normal
from sklearn.mixture import GMM
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
        * duration: DataStore (GMM) (seconds)
        * power: DataStore (Normal)
        * slope: DataStore (GMM)
        * intercept (float)
        * spike_histogram: 2D DataStore (GMM), one col per bin 
          (don't bother recording bin edges, assume these remain constant
           in fact, put bin edges in a config.py file)
        * count_per_run = DataStore (GMM): number of times this power state is seen per run 
        * current_count_per_run (int)

    """
    def __init__(self,  **kwds):
        super(PowerState, self).__init__(**kwds)

    def prepare_for_power_state_graph(self):
        new = copy.copy(self)

        new.count_per_run = DataStore(model=GMM())
        new.current_count_per_run = 1

        duration = (self.end - self.start).total_seconds()
        new.duration = DataStore(model=GMM())
        new.duration.append(duration)

        new.slope = DataStore(model=GMM())
        new.slope.append(self.slope)

        new.intercept = DataStore(model=GMM())
        new.intercept.append(self.intercept)

        new.spike_histogram = DataStore(n_columns=7, model=GMM())
        new.spike_histogram.append(self.spike_histogram)

        del new.start
        del new.end

        return new

    def save_count_per_run(self):
        self.count_per_run.append(np.array([self.current_count_per_run]))
        self.current_count_per_run = 0

    def similar(self, other):
        return self.power.model.similar_mean(other.power.model)

    def merge(self, other):
        """Merges ``other`` into ``self``."""
        self.current_count_per_run += 1
        for attribute in ['duration', 'power', 'slope', 'intercept', 
                          'spike_histogram', 'count_per_run']:
            eval('self.{attr}.extend(other.{attr})'.format(attr=attribute))

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
