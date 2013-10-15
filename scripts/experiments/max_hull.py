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
#SIG_DATA_FILENAME = 'washingmachine1.csv'

chan = Channel()
chan.load_wattsup(path.join(DATA_DIR, SIG_DATA_FILENAME))

fig = plt.figure()
ax = fig.add_subplot(111)
chan.plot(ax=ax, color='grey', date_format='%H:%M')
ax.set_xlabel('hours:minutes')

WINDOW = 40
rolling_max = pd.rolling_max(chan.series, window=WINDOW, center=True)
rolling_max = rolling_max.dropna().astype(np.float32)
rolling_max.plot(ax=ax)
steady_states = fd.steady_states(rolling_max, min_n_samples=WINDOW+5)
plot_steady_states(ax, steady_states)

plt.show()
