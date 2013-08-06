#!/usr/bin/env python

from __future__ import print_function, division
from pda.channel import Channel, DD    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import slicedpy.feature_detectors as fd

DATA_DIR = DD
c = Channel(DATA_DIR, 'aggregate')
c = c.crop('2013/6/1', '2013/6/5')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(c.series, color='k', label='data')

WINDOW_SIZE = 10

########################
# calculate rolling mean
# ax.plot(pd.rolling_mean(c.series, WINDOW_SIZE, center=True), label='mean')
# ax.plot(pd.rolling_median(c.series, WINDOW_SIZE, center=True), label='median')
# ax.plot(pd.rolling_std(c.series, WINDOW_SIZE, center=True), label='std')

##################
# Calculate low-pass
# b, a = signal.butter(8, 0.15)
# y = signal.filtfilt(b, a, c.series.values)
# ax.plot(y, label='lowpass')

#####################
# My steady state code

RDT = 0.05 # relative deviation threshold

def mean_relative_deviation(x, mean):
    """Convert to absolute value only after calculating the mean.
    The idea is that rapid oscillations should cancel themselves out."""
    return np.fabs((x - mean).mean() / mean)

n_chunks = int(c.series.size / WINDOW_SIZE)
print("n_chunks =", n_chunks)
ss_start_i = 0
steady_states = []
for chunk_i in range(n_chunks-2):
    ss_end_i = (chunk_i+1)*WINDOW_SIZE
    ss = c.series.values[ss_start_i:ss_end_i]
    next_chunk = c.series.values[ss_end_i:(chunk_i+2)*WINDOW_SIZE]
    if mean_relative_deviation(next_chunk, ss.mean()) > RDT:
        # new chunk marks the end of the steady state
        if (ss_end_i - ss_start_i) / WINDOW_SIZE > 1:
            steady_states.append((ss_start_i, ss_end_i, ss.mean()))
        ss_start_i = ss_end_i

print(len(steady_states))

for ss_start_i, ss_end_i, ss_mean in steady_states:
    line, = ax.plot([ss_start_i, ss_end_i], [ss_mean, ss_mean], color='r', linewidth=5, alpha=0.5) 
line.set_label('my steady states')

#######################
# Hart's steady states
steady_states = fd.steady_states(c.series.values)
for ss in steady_states:
    line, = ax.plot([ss.start, ss.end], [ss.mean, ss.mean], color='g', linewidth=5, alpha=0.5) 
line.set_label('Hart\'s steady states')

plt.legend()
plt.show()
