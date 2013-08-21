#!/usr/bin/env python

from pda.dataset import init_aggregate_and_appliance_dataset_figure
import matplotlib.pyplot as plt
from scipy.stats import *
import numpy as np

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4 10:00', end_date='2013/6/4 13:30',
    n_subplots=2, date_format='%H:%M:%S', alpha=0.6, 
    plot_appliance_ground_truth=False)

DISPLAY = ['mean', 'std', 'ptp', 'gmean', 'skew']

WINDOW = 60
n = chan.series.size - WINDOW
labels = ['mean', 'std', 'ptp', 'gmean', 'skew']
summary_stats = np.empty((n,len(labels)))

print("Calculating...")
for i in range(1,n):
    chunk = chan.series.values[i:i+WINDOW]
    summary_stats[i] = (chunk.mean(), chunk.std(), chunk.ptp(),
                        gmean(chunk), skew(chunk))

print("Plotting...")
for i, label in enumerate(labels):
    if label in DISPLAY:
        subplots[1].plot(chan.series.index[WINDOW:], summary_stats[:,i], 
                         label=label)

plt.legend()
plt.grid()
plt.show()
print("Done!")
