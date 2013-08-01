#!/usr/bin/env python

from pda.channel import Channel, DD    
import slicedpy.feature_detectors as fd
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = DD
c = Channel(DATA_DIR, 'aggregate')
c = c.crop('2013/6/1', '2013/6/5')

steady_states = fd.steady_states(c.series.values)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(c.series, color='k')

for ss in steady_states:
    ax.plot([ss.start, ss.end], [ss.mean, ss.mean], color='r')

for win_type in ['boxcar', 'triang', 'blackman', 'hamming', 'bartlett',
                 'parzen',
                 'bohman',
                 'blackmanharris',
                 'nuttall',
                 'barthann']:
    smoothed = pd.rolling_window(c.series, 20, win_type, center=True)# .astype(fd.PW_DTYPE)
    # ss_from_smoothed = fd.steady_states(smoothed)
    ax.plot(smoothed, label=win_type)
#for ss in ss_from_smoothed:
#    ax.plot([ss.start, ss.end], [ss.mean, ss.mean], color='g') 

plt.legend()
plt.show()
