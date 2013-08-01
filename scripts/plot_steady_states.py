#!/usr/bin/env python

from pda.channel import Channel, DD    
import slicedpy.feature_detectors as fd
import matplotlib.pyplot as plt

DATA_DIR = DD
c = Channel(DATA_DIR, 'aggregate')
c = c.crop('2013/6/1', '2013/6/5')

steady_states = fd.steady_states(c.series.values)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(c.series, color='k')

for ss in steady_states:
    ax.plot([ss.start, ss.end], [ss.mean, ss.mean], color='r')

plt.show()
