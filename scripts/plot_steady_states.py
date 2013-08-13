#!/usr/bin/env python

from pda.dataset import init_aggregate_and_appliance_dataset_figure
import slicedpy.feature_detectors as fd
from slicedpy.plot import plot_steady_states
import matplotlib.pyplot as plt
import pandas as pd

SMOOTHING = False

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4', end_date='2013/6/4 18:00',
    n_subplots=3 if SMOOTHING else 2,
    date_format='%H:%M:%S', alpha=0.6)

########################
# HART'S STEADY STATES
print("Hart's steady states...")
steady_states = fd.steady_states(chan.series.values)
plot_steady_states(subplots[0], steady_states, chan.series.index, 
                   color='b', label='Hart')

#####################
# MEAN STEADY STATES
print("Mean steady states...")
mean_steady_states = fd.mean_steady_states(chan.series.values,
                                           max_range=15)
plot_steady_states(subplots[0], mean_steady_states, chan.series.index,
                   offset=1, color='g', label='Mean')

#####################
# SLIDING MEANS STEADY STATES
print("Sliding mean steady states...")
sliding_mean_steady_states = fd.sliding_mean_steady_states(chan.series.values,
                                                           max_range=15)
plot_steady_states(subplots[0], sliding_mean_steady_states, chan.series.index,
                   offset=2, color='y', label='Sliding mean')

#####################
# RELATIVE DEVIATION STEADY STATES
print("Relative deviation steady states...")
relative_deviation_steady_states = fd.relative_deviation_steady_states(chan.series.values)
plot_steady_states(subplots[0], relative_deviation_steady_states, 
                   chan.series.index, offset=-1, color='c', label='Relative deviation')

####################
subplots[0].legend()

######################
# SMOOTHING
if SMOOTHING:
    print("Smoothing...")
    smoothed = pd.rolling_mean(chan.series, 20, center=True).dropna().astype(fd.PW_DTYPE)
    subplots[2].plot(smoothed.index, smoothed.values, label='smoothed', color='k')
    ss_from_smoothed = fd.steady_states(smoothed.values)
    plot_steady_states(subplots[2], ss_from_smoothed, color='b', index=smoothed.index)
    subplots[2].legend()

plt.show()
