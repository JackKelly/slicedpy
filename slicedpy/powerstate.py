from __future__ import print_function, division
from bunch import Bunch
import copy
from sklearn.mixture import GMM
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pda.channel import DEFAULT_TIMEZONE
from slicedpy.normal import Normal
from slicedpy.datastore import DataStore
import feature_detectors as fd

class PowerSuper(Bunch):
    def __init__(self, **kwds):
        self.duration = None
        self.power = None
        self.slope = None
        self.intercept = None
        self.spike_histogram = None
        super(PowerSuper, self).__init__(**kwds)

    def __str__(self):
        s = ""
        if self.power is not None:
            model = self.power.get_model()
            s += "power={:.1f}W\n".format(model.mean)
            s += "std={:.1f}\n".format(model.std)
            s += "min={:.1f}\n".format(model.min)
            s += "max={:.1f}\n".format(model.max)
            s += "size={:.1f}\n".format(model.size)
        return s

class PowerState(PowerSuper):
    """
    A washing machine might have three power states: washing, heating,
    spinning.

    Attributes:
        * duration: DataStore (GMM) (seconds)
        * power: DataStore (Normal)
        * slope: DataStore (GMM)
        * intercept (float)
        * spike_histogram: 2D DataStore (GMM), one col per bin 
        * count_per_run = DataStore (GMM): number of times this power state is 
          seen per run 
        * current_count_per_run (int)

    """

    def __init__(self, other=None, name='', **kwds):
        super(PowerState, self).__init__(**kwds)

        # "cast" from PowerSegment...
        if isinstance(other, PowerSegment):
            self.power = other.power
            self.count_per_run = DataStore(model=GMM())
            self.current_count_per_run = 1
            self.essential = None

            other.duration = (other.end - other.start).total_seconds()

            # Convert from scalars to DataStores:
            for attr, n_columns in [('duration', 1), 
                                    ('slope', 1), 
                                    ('intercept', 1),
                                    ('spike_histogram', 8)]:
                if other.__dict__.get(attr) is not None:
                    self.__dict__[attr] = DataStore(n_columns=n_columns, model=GMM())
                    self.__dict__[attr].append(other.__dict__[attr])

    def configure_as_off(self):
        """Configure this PowerState as 'off'."""
        self.power = DataStore(model=Normal())
        self.power.append(0)
        self.count_per_run = DataStore(model=GMM())
        self.current_count_per_run = 1
        self.essential = None
        self.end = pd.Timestamp('1970-01-01 00:00:00+00:00', tz=DEFAULT_TIMEZONE)

    def get_feature_vector(self):
        fv = [self.duration.data[0]]

        if self.slope is None:
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

    def similar(self, other, mode='min max', plus_minus=50):
        """
        Args:
          mode (str): 'min max' | 'plus minus' | 'ttest'
          'min max': uses fd.min_max_power_segments to decide if power segments
                     are similar or not.
        """
        if mode == 'min max':
            concdata = np.concatenate([self.power.data, other.power.data])
            pwr_segs = fd.min_max_power_sgmnts(concdata)
            return len(pwr_segs) == 1
        elif mode == 'plus minus':
            own_mean = self.power.get_model().mean
            other_mean = other.power.get_model().mean
            return own_mean - plus_minus < other_mean < own_mean + plus_minus
        elif mode == 'ttest':
            return self.power.get_model().similar_mean(other.power.get_model())
        else:
            raise Exception('unrecognised mode.')

    def merge(self, other):
        """Merges ``other`` into ``self``."""
        print("Merging {:.2f}W".format(self.power.get_model().mean))
        self.current_count_per_run += 1
        for attribute in ['duration', 'power', 'slope', 'intercept', 
                          'spike_histogram', 'count_per_run']:
            if self.__dict__[attribute] is not None:
                try:
                    self.__dict__[attribute].extend(other.__dict__[attribute])
                except KeyError:
                    # Not all powerstates have every attribute.
                    pass


class PowerSegment(PowerSuper):
    """
    A washing machine might have lots PowerSegments: wash, heat, wash, 
    heat, wash, spin...

    PowerSegments have start and end variables; PowerStates do not.

    Attributes:
        * start: datetime of start of each power state
        * end: datetime of end of each power state
        * duration: float, seconds
        * power: DataStore (Normal)
        * slope: float
        * intercept: float
        * spike_histogram: pd.DataFrame
          (don't bother recording bin edges, assume these remain constant
           in fact, put bin edges in a config.py file)
    """

    def __init__(self, **kwds):
        self.start = None
        self.end = None
        super(PowerSegment, self).__init__(**kwds)

    def plot(self, ax, color='k'):
        model = self.power.get_model()

        # Plot mean line
        ax.plot([self.start, self.end], 
                [model.mean, model.mean], 
                color=color, linewidth=2, alpha=0.8)

        # Plot 1 standard deviation
        num_start = mdates.date2num(self.start)
        num_end = mdates.date2num(self.end)
        width = num_end - num_start
        std_rect = plt.Rectangle(xy=(self.start, model.mean - model.std),
                                 width=width,
                                 height=(model.std * 2), 
                                 alpha=0.8, color="#aaaaaa")
        ax.add_patch(std_rect)

        # Plot min and max
        std_rect = plt.Rectangle(xy=(self.start, model.min),
                                 width=width,
                                 height=(model.max - model.min), 
                                 alpha=0.5, color="#aaaaaa")
        ax.add_patch(std_rect)
        
        # Plot slop
        if self.slope is not None:
            print("plotting slope: intercept=", self.intercept, 
                  "slope=", self.slope)
            curve = lambda x, c, m: c + (m / x)
            num_end = num_start + (10 / mdates.SEC_PER_DAY)
            X = np.linspace(num_start, num_end, 10)
            x = X * mdates.SEC_PER_DAY
            ax.plot(X, 
                    curve((x-x[0])+1, self.intercept, self.slope),
                    color=color)
