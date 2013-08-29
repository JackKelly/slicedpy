#!/usr/bin/env python

from __future__ import print_function, division
from pda.dataset import init_aggregate_and_appliance_dataset_figure
import slicedpy.feature_detectors as fd
import slicedpy.plot as splt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import DBSCAN

WINDOW_DURATION = 60 # seconds
MERGE_SPIKES = True
N_BINS = 8
BIN_I = 5 # which spike histogram column (bin) index to use in clustering?

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4', end_date='2013/6/7', n_subplots=4)

##############################################################
# High freq detector
# SPIKE HISTOGRAM

spike_histogram, bin_edges = fd.spike_histogram(chan.series, 
                                                merge_spikes=MERGE_SPIKES,
                                                window_duration=WINDOW_DURATION,
                                                n_bins=N_BINS)

splt.plot_spike_histogram(subplots[2], spike_histogram, bin_edges, 
                          title=('Merged spike histogram' if MERGE_SPIKES 
                                 else 'Spike histogram'))

##############################################################
# CLUSTERING

bin_data = spike_histogram.icol(BIN_I)

# Scale x (time) because the ordinal time representation of 1 unit 
# per day is too coarse.
scale_x = mdates.SEC_PER_DAY / WINDOW_DURATION
X = fd.spike_histogram_bin_to_data_coordinates(bin_data, 
                                               scale_x=scale_x)
db = DBSCAN(eps=10., min_samples=6).fit(X)

# inverse scaling to get back to sensible ordinal time representation
X[:,0] /= scale_x

# Plot clusters
bin_label = ('bin_data index={}, bin edges={:.0f}-{:.0f}W'
             .format(BIN_I, bin_edges[BIN_I], bin_edges[BIN_I+1]))
splt.plot_clusters(subplots[3], db, X, title_append=bin_label)

plt.show()
print('Done')
