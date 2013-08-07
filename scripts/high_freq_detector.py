#!/usr/bin/env python

from __future__ import print_function, division
from pda.channel import Channel, DD    
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = DD
WINDOW_SIZE = 10
N_BINS = 7

c = Channel(DATA_DIR, 'aggregate')
c = c.crop('2013/6/2', '2013/6/3') # 06:00')

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(np.arange(c.series.size)/WINDOW_SIZE,c.series, color='k', label='data')
ax1.set_ylabel('watts')
xlim = ax1.get_xlim()[1]
ax2 = fig.add_subplot(2,1,2, sharex=ax1)

####################
# High freq detector

fdiff = np.diff(c.series.values)
fdiff = np.fabs(fdiff)
n_chunks = int(fdiff.size / WINDOW_SIZE)
start_i = 0
bins = np.exp(np.arange(1,N_BINS+2))
print("N_BINS=", N_BINS, "n_chunks=", n_chunks)
spike_histogram = np.empty((N_BINS, n_chunks), dtype=np.float32)
for chunk_i in range(n_chunks):
    end_i = (chunk_i+1)*WINDOW_SIZE
    chunk = fdiff[start_i:end_i]
    spike_histogram[:,chunk_i] = np.histogram(chunk, bins=bins)[0]
    start_i = end_i

print(spike_histogram)

ax2.imshow(spike_histogram, aspect='auto', vmin=0, vmax=spike_histogram.max(), 
           interpolation='none', origin='lower')
ax2.set_xlim([0, xlim])
ax2.set_yticklabels(['{:.0f}'.format(x) for x in bins])
ax2.set_ylabel('Bin edges in watts')
# plt.legend()
plt.show()
