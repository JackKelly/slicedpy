#!/usr/bin/env python

from __future__ import print_function, division
from pda.channel import Channel, DD
from pda.dataset import load_dataset, crop_dataset, plot_each_channel_activity
import slicedpy.feature_detectors as fd
import slicedpy.plot as splt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pytz
from sklearn.cluster import DBSCAN

DATA_DIR = DD
WINDOW_SIZE = 60
N_SUBPLOTS = 4
START_DATE = '2013/6/4'
END_DATE = '2013/6/7'
MERGE_SPIKES = True

# SETUP FIGURE
fig = plt.figure()
fig.canvas.set_window_title(START_DATE + ' - ' + END_DATE)
ax1 = fig.add_subplot(N_SUBPLOTS,1,1)
ax2 = fig.add_subplot(N_SUBPLOTS,1,2, sharex=ax1)
ax3 = fig.add_subplot(N_SUBPLOTS,1,3, sharex=ax1)
ax4 = fig.add_subplot(N_SUBPLOTS,1,4, sharex=ax1)
#ax5 = fig.add_subplot(N_SUBPLOTS,1,5, sharex=ax1)

# LOAD AGGREGATE DATA
c = Channel()
print('Loading Current Cost aggregate...')
cc = Channel(DD, 'aggregate') # cc = Current cost
print('Loading high freq mains...')
c.load_normalised(DATA_DIR, high_freq_basename='mains.dat', 
                  high_freq_param='active')
print('Cropping power...')
c = c.crop(START_DATE, END_DATE)
cc = cc.crop(START_DATE, END_DATE)

print('Plotting...')
ax1.xaxis.axis_date(tz=pytz.timezone('Europe/London'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:%M:%S'))
ax1.plot(c.series.index, c.series, color='k', label=c.name)
ax1.plot(cc.series.index, cc.series, color='r', label=cc.name)
ax1.set_ylabel('watts')
ax1.set_title('Aggregate. 1s active power, normalised.')
ax1.legend()

# load dataset
ds = load_dataset(DATA_DIR, ignore_chans=['aggregate', 'amp_livingroom', 'adsl_router',
                                          'livingroom_s_lamp', 'gigE_&_USBhub',
                                          'livingroom_s_lamp2', 'iPad_charger', 
                                          'subwoofer_livingroom', 'livingroom_lamp_tv',
                                          'DAB_radio_livingroom', 'kitchen_lamp2',
                                          'kitchen_phone&stereo', 'utilityrm_lamp', 
                                          'samsung_charger', 'kitchen_radio', 'bedroom_chargers',
                                          'data_logger_pc', 'childs_table_lamp', 'baby_monitor_tx',
                                          'battery_charger', 'office_lamp1', 'office_lamp2',
                                          'office_lamp3', 'gigE_switch'])
                  # only_load_chans=['washing_machine', 'dishwasher', 
                  #                  'kettle', 'toaster', 'laptop', 'tv', 'htpc'])
ds = crop_dataset(ds, START_DATE, END_DATE)
plot_each_channel_activity(ax2, ds)

##############################################################
# High freq detector
# SPIKE HISTOGRAM

fdiff = np.diff(c.series.values)

if MERGE_SPIKES:
    fdiff = fd.merge_spikes(fdiff)

# ax2.plot(x[:-1], fdiff, color='k', label='fdiff')
# ax2.plot(x[:-2], merged_fdiff, color='r', label='merged_fdiff')
# ax2.plot(x[:-2], sign_comparison*1000, color='g', label='sign_comparison')

N_BINS = 8
spike_histogram, bin_edges = fd.spike_histogram(fdiff, window_size=WINDOW_SIZE, 
                                                n_bins=N_BINS)

# get ordinal representations of start and end date times
num_start_time = mdates.date2num(c.series.index[0])
num_end_time = mdates.date2num(c.series.index[-1])
splt.plot_spike_histogram(ax3, spike_histogram, bin_edges, 
                          num_start_time, num_end_time,
                          title=('Merged spike histogram' if MERGE_SPIKES 
                                 else 'Spike histogram'))


##############################################################
# CLUSTERING

ROW_I = 5
row = spike_histogram[ROW_I,:]
X = fd.spike_histogram_row_to_data_coordinates(row)
db = DBSCAN(eps=10, min_samples=6).fit(X)

##############
# Plot clusters

# calculate some constants to help convert from indicies to 
# ordinal datetimes
n_chunks = int(fdiff.size / WINDOW_SIZE)
num_time_range = num_end_time - num_start_time
num_per_item = num_time_range / n_chunks
row_label = ('row index={}, bin edges={:.0f}-{:.0f}W'
             .format(ROW_I, bin_edges[ROW_I], bin_edges[ROW_I+1]))

splt.plot_clustered_spike_histogram_row(ax4, db, X, num_start_time, num_per_item,
                                        row_label=row_label)

plt.show()
