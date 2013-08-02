#!/usr/bin/env python

from pda.channel import Channel, DD    
import matplotlib.pyplot as plt
from scipy import signal

DATA_DIR = DD
c = Channel(DATA_DIR, 'aggregate')
c = c.crop('2013/6/1', '2013/6/5')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(c.series, color='k')

# Calculate low-pass
b, a = signal.butter(8, 0.15)
y = signal.filtfilt(b, a, c.series.values)
ax.plot(y, color='b')

plt.legend()
plt.show()
