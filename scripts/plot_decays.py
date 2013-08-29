#!/usr/bin/env python

from pda.dataset import init_aggregate_and_appliance_dataset_figure
import slicedpy.feature_detectors as fd
from slicedpy.plot import plot_steady_states
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

SLIDING_MEANS_STEADY_STATES = False
RD_STEADY_STATES = True
TTESTS = False
SS_LINREGRESS = False
STD = False # spike then decay
STDP = True # spike then poly decay

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4', end_date='2013/6/5',
    n_subplots=2 + np.array([TTESTS, SS_LINREGRESS, STD, STDP, STDP]).sum(),
    date_format='%H:%M:%S', alpha=0.6)

subplot_i = 2

#####################
# SLIDING MEANS STEADY STATES
if SLIDING_MEANS_STEADY_STATES:
    print("Sliding mean steady states...")
    sliding_mean_steady_states = fd.sliding_mean_steady_states(chan.series,
                                                               max_range=15)
    plot_steady_states(subplots[0], sliding_mean_steady_states,
                       offset=2, color='y', label='Sliding mean')


#####################
# RELATIVE DEVIATION STEADY STATES
if RD_STEADY_STATES:
    print("Relative deviation steady states...")
    relative_deviation_power_segments = fd.relative_deviation_power_sgmnts(chan.series)
    plot_steady_states(subplots[0], relative_deviation_power_segments, 
                        offset=-1, color='c', label='Relative deviation')


####################
subplots[0].legend()

##################
# T Test...
if TTESTS:
    print("Calculating and plotting t-tests...")
    for start, ss in relative_deviation_power_segments.iterrows():
        end = ss['end']
        p_value = fd.ttest_both_halves(chan.series, start=start, end=end)
        subplots[subplot_i].plot([start, end], [p_value, p_value], color='r', linewidth=4)
    subplots[subplot_i].set_title('p value for both halves of steady state from relative duration')
    subplots[subplot_i].set_ylabel('p value')
    subplot_i += 1

##################
# Linear regression...
if SS_LINREGRESS:
    print("Calculating and plotting linear regression...")
    for start, ss in relative_deviation_power_segments.iterrows():
        end = ss['end']
        slope, r_value, p_value, stderr = fd.linregress(chan.series, start, end)
        subplots[subplot_i].plot([start, end], [slope, slope], color='r', linewidth=4)
    subplots[subplot_i].set_title('Slope from linear regression from relative duration steady states')
    subplots[subplot_i].set_ylabel('slope in watts/second')
    subplot_i += 1

##################
# Spike then decay
if STD:
    print("Calculating and plotting spike then decay...")
    stds = fd.spike_then_decay(chan.series, mode='linear')
    for start, std in stds.iterrows():
        end = std['end']
        subplots[subplot_i].plot([start], [std['slope']],
                                 'o', markersize=6, color='r', linewidth=4)
    subplots[subplot_i].set_title('Spike Then Decay')
    subplots[subplot_i].set_ylabel('slope in watts/second')
    subplots[subplot_i].grid()
    subplot_i += 1

######################
#
# NEEDS 2 SUBPLOTS!
if STDP:
    print("Calculating and plotting spike then poly decay...")
    stds = fd.spike_then_decay(chan.series, mode='poly')
    curve = lambda x, c, m: c + (m / x)
    for start, std in stds.iterrows():
        end = std['end']
        X = chan.series.index[(chan.series.index >= start) & 
                              (chan.series.index < end)]
        X = mdates.date2num(X)
        x = X * mdates.SEC_PER_DAY
        subplots[subplot_i].plot([start], [std['slope']],
                                 'o', markersize=6, color='r', linewidth=4)
        subplots[subplot_i+1].plot(X, 
                                   curve((x-x[0])+1, std['intercept'], 
                                         std['slope']),
                                   color='r')

    subplots[subplot_i].set_title('Spike Then poly Decay.  y = c + m/x')
    subplots[subplot_i].set_ylabel('m')
    subplots[subplot_i].grid()

    subplots[subplot_i+1].set_title('Reconstructed curve')
    subplots[subplot_i+1].set_ylabel('watts')
    subplots[subplot_i+1].grid()

    subplot_i += 2

plt.show()
