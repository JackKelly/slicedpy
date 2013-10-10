from __future__ import print_function, division
from pda.channel import Channel
from slicedpy.appliance import Appliance
import slicedpy.feature_detectors as fd
from slicedpy.plot import plot_steady_states
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import numpy as np

DATA_DIR = '/data/mine/domesticPowerData/BellendenRd/wattsUp'
SIG_DATA_FILENAME = 'breadmaker1.csv'

chan = Channel()
chan.load_wattsup(path.join(DATA_DIR, SIG_DATA_FILENAME))

fig = plt.figure()
ax = fig.add_subplot(111)
chan.plot(ax=ax, color='grey')

for window in [2, 5, 10, 50, 100, 300]:
    print(window)
    smoothed = pd.rolling_window(chan.series, window, 'blackmanharris', center=True)
#    smoothed.plot(ax=ax)
#    pd.rolling_mean(chan.series, window, center=True).plot()
    steady_states = fd.steady_states(smoothed.astype(np.float32).dropna())
    plot_steady_states(ax, steady_states, color='b', label=str(window))

plt.show()
