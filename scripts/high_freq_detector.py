#!/usr/bin/env python

from __future__ import print_function, division
from pda.channel import Channel, DD
from pda.dataset import load_dataset, crop_dataset, plot_each_channel_activity
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pytz

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

# Merge consecutive fdiff values of the same sign
sign_comparison = (fdiff[:-1] * fdiff[1:]) > 0
merged_fdiff = np.zeros(sign_comparison.size)
accumulator = 0
for i in range(1,sign_comparison.size-1):
    if sign_comparison[i] == True:
        if accumulator == 0:
            accumulator = fdiff[i] + fdiff[i+1]
        else:
            accumulator += fdiff[i+1]
    else:
        if accumulator == 0:
            merged_fdiff[i] = fdiff[i]
        else:
            merged_fdiff[i] = accumulator
            accumulator = 0

# ax2.plot(x[:-1], fdiff, color='k', label='fdiff')
# ax2.plot(x[:-2], merged_fdiff, color='r', label='merged_fdiff')
# ax2.plot(x[:-2], sign_comparison*1000, color='g', label='sign_comparison')

abs_fdiff = np.fabs(merged_fdiff if MERGE_SPIKES else fdiff)
n_chunks = int(abs_fdiff.size / WINDOW_SIZE)
start_i = 0
N_BINS = 8
bin_edges = np.concatenate(([0], np.exp(np.arange(1,N_BINS+1))))
print("bin edges =", bin_edges)
# bin_edges = [5, 10, 15, 20, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 4000]
# N_BINS = len(bin_edges)-1
print("N_BINS=", N_BINS, "n_chunks=", n_chunks)
spike_histogram = np.empty((N_BINS, n_chunks), dtype=np.int64)
for chunk_i in range(n_chunks):
    end_i = (chunk_i+1)*WINDOW_SIZE
    chunk = abs_fdiff[start_i:end_i]
    spike_histogram[:,chunk_i] = np.histogram(chunk, bins=bin_edges)[0]
    start_i = end_i

vmax = spike_histogram.max()

# get ordinal representations of start and end date times
num_start_time = mdates.date2num(c.series.index[0])
num_end_time = mdates.date2num(c.series.index[-1])

ax3.imshow(spike_histogram, aspect='auto', vmin=0, vmax=vmax, 
           interpolation='none', origin='lower',
           extent=(num_start_time, num_end_time, 0, N_BINS))
                                
ax3.set_ylabel('Bin edges in watts')
ax3.set_title(('Merged' if MERGE_SPIKES else 'Unmerged') + ' spike histogram')

def bin_label(bin_i, pos=None):
    bin_i = int(bin_i)
    if bin_i >= len(bin_edges)-1:
        bin_i = len(bin_edges)-2
    return ('bin index={}, bin edges={:.0f}-{:.0f}W'
            .format(bin_i, bin_edges[bin_i], bin_edges[bin_i+1]))

ax3.set_yticks(np.arange(N_BINS+1))
ax3.set_yticklabels(['{:.0f}'.format(w) for w in bin_edges])

##############################################################
# CLUSTERING

from sklearn.cluster import DBSCAN
import pylab as pl

BIN_I = 5

bin = spike_histogram[BIN_I,:]
# make a 2d matrix X where each row stores the 
# coordinates of a single data point.
nonzero_i = np.nonzero(bin)[0]
X = np.zeros((nonzero_i.size, 2), dtype=np.float64)
X[:,0] = nonzero_i
X[:,1] = bin[nonzero_i]
db = DBSCAN(eps=10, min_samples=6).fit(X)

labels = db.labels_
core_samples = db.core_sample_indices_
unique_labels = set(labels)
colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# calculate some constants to help convert from indicies to 
# ordinal datetimes
num_time_range = num_end_time - num_start_time
num_per_item = num_time_range / n_chunks
def i_to_num(i):
    return num_start_time + (i*num_per_item)

for k, col in zip(unique_labels, colors):
    if k == -1:
        # black used for noise
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    for index in class_members:
        x = X[index]
        if index in core_samples and k != -1:
            markersize = 14
        else:
            markersize = 6
        ax4.plot(i_to_num(x[0]), x[1], 'o', markerfacecolor=col, 
                 markeredgecolor='k', markersize=markersize)    

ax4.set_title('clustering using DBSCAN from ' + 
              ('merged' if MERGE_SPIKES else 'unmerged') +
              ', eps={}, min_samples={}, '
              '{}'
              .format(db.eps, db.min_samples, bin_label(BIN_I)))
ax4.set_ylabel('count')
ax4.set_xlabel('date time')

plt.show()
