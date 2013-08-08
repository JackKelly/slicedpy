#!/usr/bin/env python

from __future__ import print_function, division
from pda.channel import Channel, DD
from pda.dataset import load_dataset, crop_dataset, dataset_to_dataframe, plot_each_channel_activity
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

DATA_DIR = DD
WINDOW_SIZE = 10
N_SUBPLOTS = 4
START_DATE = '2013/6/2'
END_DATE = '2013/6/4'

c = Channel(DATA_DIR, 'aggregate')
c = c.crop(START_DATE, END_DATE)
c.series = c.series.resample('6S', how='max')

fig = plt.figure()
fig.canvas.set_window_title(START_DATE + ' - ' + END_DATE)
ax1 = fig.add_subplot(N_SUBPLOTS,1,1)
x = np.arange(c.series.size)/WINDOW_SIZE

#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
#ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.plot(c.series.index, c.series, color='k', label='data')
ax1.xaxis.axis_date()
#fig.autofmt_xdate()

ax1.set_ylabel('watts')
ax1.set_title('Aggregate.  Resampled to 6seconds')
xlim = ax1.get_xlim()[1]
ax2 = fig.add_subplot(N_SUBPLOTS,1,2, sharex=ax1)
ax3 = fig.add_subplot(N_SUBPLOTS,1,3, sharex=ax1)
ax4 = fig.add_subplot(N_SUBPLOTS,1,4, sharex=ax1)
#ax5 = fig.add_subplot(N_SUBPLOTS,1,5, sharex=ax1)
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
ax2.xaxis.axis_date()
plot_each_channel_activity(ax2, ds)

raise SystemExit(0)

####################
# High freq detector

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

abs_fdiff = np.fabs(merged_fdiff)
n_chunks = int(abs_fdiff.size / WINDOW_SIZE)
start_i = 0
N_BINS = 7
bins = np.exp(np.arange(1,N_BINS+2))
# bins = [5, 10, 15, 20, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 4000]
# N_BINS = len(bins)-1
print("N_BINS=", N_BINS, "n_chunks=", n_chunks)
spike_histogram = np.empty((N_BINS, n_chunks), dtype=np.int64)
for chunk_i in range(n_chunks):
    end_i = (chunk_i+1)*WINDOW_SIZE
    chunk = abs_fdiff[start_i:end_i]
    spike_histogram[:,chunk_i] = np.histogram(chunk, bins=bins)[0]
    start_i = end_i

vmax = spike_histogram.max()

ax3.imshow(spike_histogram, aspect='auto', vmin=0, vmax=vmax, 
           interpolation='none', origin='lower')
ax3.set_xlim([0, xlim])
ax3.set_yticklabels(['{:.0f}'.format(w) for w in bins])
ax3.set_ylabel('Bin edges in watts')
ax3.set_title('Merged spike histogram')


from sklearn.cluster import DBSCAN
import pylab as pl

row = spike_histogram[1,:]
# make a 2d matrix X where each row stores the 
# coordinates of a single data point.
nonzero_i = np.nonzero(row)[0]
X = np.zeros((nonzero_i.size, 2), dtype=np.float64)
X[:,0] = nonzero_i
X[:,1] = row[nonzero_i]
db = DBSCAN(eps=10, min_samples=6).fit(X)

labels = db.labels_
core_samples = db.core_sample_indices_
unique_labels = set(labels)
colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
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
        ax4.plot(x[0], x[1], 'o', markerfacecolor=col, 
                 markeredgecolor='k', markersize=markersize)

ax4.set_title('clustering using DBSCAN from merged, eps={}, min_samples={}'
              .format(db.eps, db.min_samples))
ax4.set_ylabel('count')
ax4.set_xlabel('minutes')



# abs_fdiff = np.fabs(fdiff)
# n_chunks = int(abs_fdiff.size / WINDOW_SIZE)
# start_i = 0
# print("N_BINS=", N_BINS, "n_chunks=", n_chunks)
# spike_histogram = np.empty((N_BINS, n_chunks), dtype=np.int64)
# for chunk_i in range(n_chunks):
#     end_i = (chunk_i+1)*WINDOW_SIZE
#     chunk = abs_fdiff[start_i:end_i]
#     spike_histogram[:,chunk_i] = np.histogram(chunk, bins=bins)[0]
#     start_i = end_i

# ax4.imshow(spike_histogram, aspect='auto', vmin=0, vmax=vmax, 
#            interpolation='none', origin='lower')
# ax4.set_xlim([0, xlim])
# ax4.set_yticklabels(['{:.0f}'.format(w) for w in bins])
# ax4.set_ylabel('Bin edges in watts')
# ax4.set_title('Spike histogram')




# plt.legend()
plt.show()
