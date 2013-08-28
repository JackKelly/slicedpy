#!/usr/bin/env python

from __future__ import print_function, division
from pda.dataset import init_aggregate_and_appliance_dataset_figure
import slicedpy.feature_detectors as fd
import slicedpy.plot as splt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.cluster import DBSCAN

WINDOW_DURATION = 'T'
MERGE_SPIKES = True
N_BINS = 8
ROW_I = 5 # which spike histogram row (bin) index to use in clustering?

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4', end_date='2013/6/7', n_subplots=4)

##############################################################
# High freq detector
# SPIKE HISTOGRAM

spike_histogram, bin_edges = fd.spike_histogram(chan.series, 
                                                merge_spikes=MERGE_SPIKES,
                                                window_duration=WINDOW_DURATION, 
                                                n_bins=N_BINS)

# get ordinal representations of start and end date times
num_start_time = mdates.date2num(chan.series.index[0])
num_end_time = mdates.date2num(chan.series.index[-1])
splt.plot_spike_histogram(subplots[2], spike_histogram, bin_edges, 
                          num_start_time, num_end_time,
                          title=('Merged spike histogram' if MERGE_SPIKES 
                                 else 'Spike histogram'))

##############################################################
# CLUSTERING

row = spike_histogram[ROW_I,:]
X = fd.spike_histogram_row_to_data_coordinates(row)
db = DBSCAN(eps=10, min_samples=6).fit(X)

##############
# Plot clusters

# calculate some constants to help convert from indicies to 
# ordinal datetimes
s_duration = (chan.series.index[-1] - chan.series.index[0]).total_seconds()
n_chunks = int(s_duration / WINDOW_DURATION)
num_time_range = num_end_time - num_start_time
num_per_item = num_time_range / n_chunks
row_label = ('row index={}, bin edges={:.0f}-{:.0f}W'
             .format(ROW_I, bin_edges[ROW_I], bin_edges[ROW_I+1]))
i_to_num = np.vectorize(lambda i: num_start_time + (i*num_per_item))

# Create new X_datetime where the the X coordinates 
# are valid ordinal datetime numbers
X_datetime = X.copy()
X_datetime[:,0] = i_to_num(X[:,0])

splt.plot_clusters(subplots[3], db, X_datetime, title_append=row_label)

plt.show()
print('Done')
