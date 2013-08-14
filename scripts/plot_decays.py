#!/usr/bin/env python

from pda.dataset import init_aggregate_and_appliance_dataset_figure
import slicedpy.feature_detectors as fd
from slicedpy.plot import plot_steady_states
import matplotlib.pyplot as plt
import numpy as np

SLIDING_MEANS_STEADY_STATES = False
RD_STEADY_STATES = False
TTESTS = False
SS_LINREGRESS = False
STD = True # spike then decay

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4', end_date='2013/6/5',
    n_subplots=2 + np.array([TTESTS, SS_LINREGRESS, STD]).sum(),
    date_format='%H:%M:%S', alpha=0.6)

subplot_i = 2

#####################
# SLIDING MEANS STEADY STATES
if SLIDING_MEANS_STEADY_STATES:
    print("Sliding mean steady states...")
    sliding_mean_steady_states = fd.sliding_mean_steady_states(chan.series.values,
                                                               max_range=15)
    plot_steady_states(subplots[0], sliding_mean_steady_states, chan.series.index,
                       offset=2, color='y', label='Sliding mean')

#####################
# RELATIVE DEVIATION STEADY STATES
if RD_STEADY_STATES:
    print("Relative deviation steady states...")
    relative_deviation_steady_states = fd.relative_deviation_steady_states(chan.series.values)
    plot_steady_states(subplots[0], relative_deviation_steady_states, 
                       chan.series.index, offset=-1, color='c', label='Relative deviation')

####################
subplots[0].legend()

##################
# T Test...
if TTESTS:
    print("Calculating and plotting t-tests...")
    for ss in relative_deviation_steady_states:
        start = chan.series.index[ss.start]
        end = chan.series.index[ss.end]
        p_value = ss.ttest_both_halves(chan.series.values)
        subplots[subplot_i].plot([start, end], [p_value, p_value], color='r', linewidth=4)
    subplots[subplot_i].set_title('p value for both halves of steady state from relative duration')
    subplots[subplot_i].set_ylabel('p value')
    subplot_i += 1

##################
# Linear regression...
if SS_LINREGRESS:
    print("Calculating and plotting linear regression...")
    for ss in relative_deviation_steady_states:
        start = chan.series.index[ss.start]
        end = chan.series.index[ss.end]
        ss.linregress(chan.series)
        subplots[subplot_i].plot([start, end], [ss.slope, ss.slope], color='r', linewidth=4)
    subplots[subplot_i].set_title('Slope from linear regression from relative duration steady states')
    subplots[subplot_i].set_ylabel('slope in watts/second')
    subplot_i += 1

##################
# Spike then decay
if STD:
    print("Calculating and plotting spike then decay...")
    stds = fd.spike_then_decay(chan.series)
    for std in stds:
        start = chan.series.index[std.start]
        end = chan.series.index[std.end]
        subplots[subplot_i].plot([start], [std.slope], 'o', markersize=6, color='r', linewidth=4)
    subplots[subplot_i].set_title('Spike Then Decay')
    subplots[subplot_i].set_ylabel('slope in watts/second')
    subplots[subplot_i].grid()
    subplot_i += 1

plt.show()
