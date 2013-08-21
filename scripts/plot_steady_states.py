#!/usr/bin/env python

from pda.dataset import init_aggregate_and_appliance_dataset_figure
import slicedpy.feature_detectors as fd
from slicedpy.plot import plot_steady_states
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import *

SMOOTHING = False

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4 11:00', end_date='2013/6/4 12:00',
    n_subplots=3 if SMOOTHING else 2,
    date_format='%H:%M:%S', alpha=0.6)

########################
# HART'S STEADY STATES
if False:
    print("Hart's steady states...")
    steady_states = fd.steady_states(chan.series.values)
    plot_steady_states(subplots[0], steady_states, chan.series.index, 
                       color='b', label='Hart')

#####################
# MEAN STEADY STATES
if False:
    print("Mean steady states...")
    mean_steady_states = fd.mean_steady_states(chan.series.values,
                                               max_range=15)
    plot_steady_states(subplots[0], mean_steady_states, chan.series.index,
                       offset=1, color='g', label='Mean')

#####################
# SLIDING MEANS STEADY STATES
if False:
    print("Sliding mean steady states...")
    sliding_mean_steady_states = fd.sliding_mean_steady_states(chan.series.values,
                                                               max_range=15)
    plot_steady_states(subplots[0], sliding_mean_steady_states, chan.series.index,
                       offset=2, color='y', label='Sliding mean')

#####################
# RELATIVE DEVIATION STEADY STATES
if False:
    print("Relative deviation steady states...")
    relative_deviation_steady_states = fd.relative_deviation_steady_states(chan.series.values)
    plot_steady_states(subplots[0], relative_deviation_steady_states, 
                       chan.series.index, offset=-1, color='c', label='Relative deviation')


#####################
# MIN MAX STEADY STATES
if True:
    print("Min-Max steady states...")
    min_max_steady_states = fd.min_max_steady_states(chan.series.values)
    plot_steady_states(subplots[0], min_max_steady_states, 
                       chan.series.index, offset=-2, color='r', label='Min Max')


#####################
# MIN MAX CHUNK STEADY STATES
if False:
    print("Mean chunk steady states...")
    mean_chunk_steady_states = fd.mean_chunk_steady_states(chan.series.values) 
    plot_steady_states(subplots[0], mean_chunk_steady_states, 
                       chan.series.index, offset=-3, color='orange', label='Mean Chunk')


####################
subplots[0].legend()

######################
# SMOOTHING
if SMOOTHING:
    print("Smoothing...")
    smoothed = pd.rolling_mean(chan.series, 20, center=True).dropna().astype(fd.PW_DTYPE)
    subplots[3].plot(smoothed.index, smoothed.values, label='smoothed', color='k')
    ss_from_smoothed = fd.steady_states(smoothed.values)
    plot_steady_states(subplots[3], ss_from_smoothed, color='b', index=smoothed.index)
    subplots[3].legend()

plt.show()
