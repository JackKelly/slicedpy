#!/usr/bin/env python

from __future__ import print_function, division
from pda.channel import Channel, DD    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

DATA_DIR = DD
c = Channel(DATA_DIR, 'aggregate')
c = c.crop('2013/6/1', '2013/6/5')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(c.series, color='k', label='data')

WINDOW_SIZE = 20

# calculate rolling mean
# ax.plot(pd.rolling_mean(c.series, WINDOW_SIZE, center=True), label='mean')
# ax.plot(pd.rolling_median(c.series, WINDOW_SIZE, center=True), label='median')
# ax.plot(pd.rolling_std(c.series, WINDOW_SIZE, center=True), label='std')

# mean absolute deviation

# mad = pd.rolling_apply(c.series, WINDOW_SIZE, mad_func, center=True)
# ax.plot(mad, color='g')

# Calculate low-pass
# b, a = signal.butter(8, 0.15)
# y = signal.filtfilt(b, a, c.series.values)
# ax.plot(y, label='lowpass')

# My algorithm:
# 1. Take the first WINDOW_SIZE chunk. Calculate the mean and MAD.
# 2. If the MAD is above threshold then move onto the next chunk else:
# 3. Look at the next WINDOW_SIZE chunk and calculate the MAD for that chunk, 
#    using the mean from the previous chunk.  If the MAD is above threshold then
#    end previous steady state and start new one.  Else add new chunk to stead state.

def mean_relative_deviation(x, mean):
    return np.fabs((x - mean).mean() / mean)

def mean_absolute_relative_deviation(x, mean):
    return np.fabs(x - mean).mean() / mean

n_chunks = int(c.series.size / WINDOW_SIZE)
print("n_chunks =", n_chunks)
ss_start_i = 0
RDT = 0.1 # relative deviation threshold
steady_states = []
is_first_chunk_of_ss = True
for chunk_i in range(n_chunks-2):
    ss_end_i = (chunk_i+1)*WINDOW_SIZE
    ss = c.series.values[ss_start_i:ss_end_i]

    if is_first_chunk_of_ss:
        if mean_absolute_relative_deviation(ss, ss.mean()) > RDT:
            ss_start_i = ss_end_i
            continue
        else:
            is_first_chunk_of_ss = False

    next_chunk = c.series.values[ss_end_i:(chunk_i+2)*WINDOW_SIZE]
    if mean_relative_deviation(next_chunk, ss.mean()) > RDT:
        # new chunk marks the end of the steady state
        steady_states.append((ss_start_i, ss_end_i, ss.mean()))
        ss_start_i = ss_end_i
        is_first_chunk_of_ss = True

print(len(steady_states))

for ss_start_i, ss_end_i, ss_mean in steady_states:
    ax.plot([ss_start_i, ss_end_i], [ss_mean, ss_mean], color='r') 

plt.legend()
plt.show()
