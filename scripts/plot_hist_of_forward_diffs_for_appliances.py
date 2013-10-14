#!/usr/bin/env python

from __future__ import print_function, division
from pda.channel import Channel
import slicedpy.feature_detectors as fd
import slicedpy.plot as splt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import DBSCAN
from os import path

WINDOW_DURATION = 60 # seconds
MERGE_SPIKES = True
N_BINS = 8
BIN_I = 6 # which spike histogram column (bin) index to use in clustering?

DATA_DIR = '/data/mine/domesticPowerData/BellendenRd/wattsUp'
SIG_DATA_FILENAME = 'breadmaker1.csv'
#SIG_DATA_FILENAME = 'washingmachine1.csv'

chan = Channel()
chan.load_wattsup(path.join(DATA_DIR, SIG_DATA_FILENAME))

fig = plt.figure()
ax = fig.add_subplot(3,1,1)
chan.plot(ax=ax, color='grey')


##############################################################
# High freq detector
# SPIKE HISTOGRAM

spike_histogram, bin_edges = fd.spike_histogram(chan.series, 
                                                merge_spikes=MERGE_SPIKES,
                                                window_duration=WINDOW_DURATION,
                                                n_bins=N_BINS)

ax2 = fig.add_subplot(3,1,2, sharex=ax)
splt.plot_spike_histogram(ax2, spike_histogram, bin_edges, 
                          title=('Merged spike histogram' if MERGE_SPIKES 
                                 else 'Spike histogram'))

##############################################################
# CLUSTERING

#bin_data = spike_histogram.icol(BIN_I)
bin_data = spike_histogram.iloc[:,1:].sum(axis=1)

# Scale x (time) because the ordinal time representation of 1 unit 
# per day is too coarse.
scale_x = mdates.SEC_PER_DAY / WINDOW_DURATION
X = fd.spike_histogram_bin_to_data_coordinates(bin_data, 
                                               scale_x=scale_x,
                                               binary_items=True)
db = DBSCAN(eps=3, min_samples=3).fit(X)

# inverse scaling to get back to sensible ordinal time representation
X[:,0] /= scale_x

# Plot clusters
#bin_label = ('bin_data index={}, bin edges={:.0f}-{:.0f}W'
#             .format(BIN_I, bin_edges[BIN_I], bin_edges[BIN_I+1]))
bin_label = 'all bins (except 0-3W) summed'
ax3 = fig.add_subplot(3,1,3, sharex=ax)
splt.plot_clusters(ax3, db, X, title_append=bin_label)

plt.show()
print('Done')
