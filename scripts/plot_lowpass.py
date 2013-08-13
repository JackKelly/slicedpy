#!/usr/bin/env python

from pda.dataset import init_aggregate_and_appliance_dataset_figure
import matplotlib.pyplot as plt
from scipy import signal

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/1', end_date='2013/6/5', n_subplots=2)

# Calculate low-pass
print("Filtering...")
N = 8
Wn = 0.15
b, a = signal.butter(N, Wn)
y = signal.filtfilt(b, a, chan.series.values)
subplots[0].plot(chan.series.index, y, color='b', 
                 label='low pass, N={:d}, Wn={:.2f}'.format(N, Wn))
subplots[0].legend()

plt.show()
print("Done.")
