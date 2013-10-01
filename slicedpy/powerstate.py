from __future__ import print_function, division
from bunch import Bunch
import copy
from slicedpy.normal import Normal
from sklearn.mixture import GMM
from slicedpy.datastore import DataStore
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pda.channel import DEFAULT_TIMEZONE

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
    def __init__(self, preset=None, name='',  **kwds):
        super(PowerState, self).__init__(**kwds)
        if preset == 'off':
            self.configure_as_off()

    def configure_as_off(self):
        """Configure this PowerState as 'off'."""
        self.power = DataStore(model=Normal())
        self.power.append(0)
        self.count_per_run = DataStore(model=GMM())
        self.current_count_per_run = 1
        self.essential = None
        self.end = pd.Timestamp('1970-01-01 00:00:00+00:00', tz=DEFAULT_TIMEZONE)

    def prepare_for_power_state_graph(self):
        new = copy.copy(self)

        new.count_per_run = DataStore(model=GMM())
        new.current_count_per_run = 1
        new.essential = None

        self.duration = (self.end - self.start).total_seconds()
        del new.start
        del new.end

        # Convert from scalars to DataStores:
        for attr, n_columns in [('duration', 1), 
                                ('slope', 1), 
                                ('intercept', 1),
                                ('spike_histogram', 8)]:

            if self.__dict__.get(attr) is not None:
                new.__dict__[attr] = DataStore(n_columns=n_columns, model=GMM())
                new.__dict__[attr].append(self.__dict__[attr])

        return new

    def get_feature_vector(self):
        fv = [self.duration.data[0]]

        if self.__dict__.get('slope') is None:
            fv.append(0)
        else:
            fv.append(self.slope.data[0])

        if self.spike_histogram.data.size == 0:
            fv.extend([0]*8)
        else:
            fv.extend(self.spike_histogram.data[0,:].tolist())
        return fv

    def save_count_per_run(self):
        self.count_per_run.append(np.array([self.current_count_per_run]))
        self.current_count_per_run = 0

    def similar(self, other, plus_minus=50):
        own_mean = self.power.get_model().mean
        other_mean = other.power.get_model().mean
        return own_mean - plus_minus < other_mean < own_mean + plus_minus
#        return self.power.get_model().similar_mean(other.power.get_model())

    def merge(self, other):
        """Merges ``other`` into ``self``."""
        print("Merging {:.2f}W".format(self.power.get_model().mean))
        self.current_count_per_run += 1
        for attribute in ['duration', 'power', 'slope', 'intercept', 
                          'spike_histogram', 'count_per_run']:
            try:
                self.__dict__[attribute].extend(other.__dict__[attribute])
            except KeyError:
                # Not all powerstates have every attribute.
                pass

    def plot(self, ax, color='k'):
        ax.plot([self.start, self.end], 
                [self.power.get_model().mean, self.power.get_model().mean], 
                color=color)
        
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

    def __str__(self):
        model = self.power.get_model()
        s = "power={:.1f}W\n".format(model.mean)
        s += "std={:.1f}\n".format(model.std)
        s += "min={:.1f}\n".format(model.min)
        s += "max={:.1f}\n".format(model.max)
        s += "size={:.1f}\n".format(model.size)
        return s
