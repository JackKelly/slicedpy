#!/usr/bin/env python
from __future__ import division, print_function
from pda.dataset import init_aggregate_and_appliance_dataset_figure
import slicedpy.feature_detectors as fd
from slicedpy.plot import plot_steady_states, plot_clusters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.cluster import DBSCAN

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4 10:00', end_date='2013/6/4 14:00',
    n_subplots=3, date_format='%H:%M:%S', alpha=0.6)

#####################
# SLIDING MEANS STEADY STATES
print("Sliding mean steady states...")
sliding_mean_steady_states = fd.sliding_mean_steady_states(chan.series,
                                                           max_range=15)
plot_steady_states(subplots[0], sliding_mean_steady_states,
                   offset=2, color='y', label='Sliding mean')


#####################
# RELATIVE DEVIATION STEADY STATES
print("Relative deviation power segments...")
relative_deviation_power_segments = fd.relative_deviation_power_sgmnts(chan.series)
plot_steady_states(subplots[0], relative_deviation_power_segments, 
                    offset=-1, color='c', label='Relative deviation')


####################
subplots[0].legend()

##############################################################
# CLUSTERING

print("Creating matrix of coordinates ready for clustering...")
SCALE_TIME = mdates.SEC_PER_DAY * 20
X = np.empty((chan.series.size, 2))
X[:,0] = mdates.date2num(chan.series.index) * SCALE_TIME
X[:,1] = chan.series.values
print("Clustering...")
db = DBSCAN(eps=300, min_samples=6).fit(X)
print("Plotting clusters...")
plot_clusters(subplots[2], db, X, scale_x=1/SCALE_TIME, 
              title_append="SCALE_TIME={}".format(SCALE_TIME))

plt.show()
print("Done!")
