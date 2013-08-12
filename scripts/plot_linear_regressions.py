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
WINDOW_SIZE = 20
N_SUBPLOTS = 2
START_DATE = '2013/6/4'
END_DATE = '2013/6/4 18:00'

# SETUP FIGURE
fig = plt.figure()
fig.canvas.set_window_title(START_DATE + ' - ' + END_DATE)
ax1 = fig.add_subplot(N_SUBPLOTS,1,1)
ax2 = fig.add_subplot(N_SUBPLOTS,1,2, sharex=ax1)
#ax3 = fig.add_subplot(N_SUBPLOTS,1,3, sharex=ax1)
#ax4 = fig.add_subplot(N_SUBPLOTS,1,4, sharex=ax1)
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

steady_states_mean = fd.steady_states_mean(c.series.values.astype(np.float32), 
                                           max_range=15)
for ssm in steady_states_mean:
    ax1.plot([c.series.index[ssm.start], c.series.index[ssm.end]], 
             [ssm.mean, ssm.mean], color='g', linewidth=4, alpha=0.8)



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

#######################
# LINEAR REGRESSIONS

#print('Calculating linear regressions...')
#mlr = fd.multiple_linear_regressions(c.series.values, window_size=WINDOW_SIZE)
#print('Plotting linear regressions...')
#splt.plot_multiple_linear_regressions(ax3, mlr, c.series.index, WINDOW_SIZE, ax4)

plt.show()
