#!/usr/bin/env python

from __future__ import print_function, division
from pda.dataset import init_aggregate_and_appliance_dataset_figure
import slicedpy.feature_detectors as fd
import slicedpy.plot as splt
import matplotlib.pyplot as plt

WINDOW_SIZE = 20 # number of samples

subplots, chan = init_aggregate_and_appliance_dataset_figure(
    start_date='2013/6/4', end_date='2013/6/4 18:00', n_subplots=4)

#######################
# LINEAR REGRESSIONS

print('Calculating linear regressions...')
mlr = fd.multiple_linear_regressions(chan.series, window_size=WINDOW_SIZE)
print('Plotting linear regressions...')
splt.plot_multiple_linear_regressions(subplots[2], mlr, 
                                      WINDOW_SIZE, subplots[3])

plt.show()
print('Done!')
