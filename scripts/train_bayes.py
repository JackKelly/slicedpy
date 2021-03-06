from __future__ import print_function, division
from pda.channel import Channel, DD
from slicedpy.appliance import Appliance
from slicedpy.disaggregator import BayesDisaggregator
from slicedpy.plot import plot_appliance_hypotheses, plot_steady_states
from pda.dataset import init_aggregate_and_appliance_dataset_figure
import matplotlib.pyplot as plt
import os.path as path

# Appliance signature data directory
APP_DATA_DIR = '/data/mine/domesticPowerData/BellendenRd/wattsUp'

# Load and plot aggregate data
# subplots, agg_chan = init_aggregate_and_appliance_dataset_figure(
#    start_date='2013/6/4', end_date='2013/6/5',
#    n_subplots=2, date_format='%H:%M:%S', alpha=0.6) #, plot_appliance_ground_truth=False)

######### WASHING MACHINE
wm_app = Appliance('wm')

wm1 = Channel()
wm1.load_wattsup(path.join(APP_DATA_DIR, 'washingmachine1.csv'))
wm_app.train_on_single_example(wm1)

wm2 = Channel()
wm2.load_wattsup(path.join(APP_DATA_DIR, 'washingmachine2.csv'))
wm_sig_power_states = wm_app.train_on_single_example(wm2)

# fig1, ax1 = plt.subplots()
# wm2.plot(ax1)
# for ps in wm_sig_power_states:
#     ps.plot(ax1)

# wm_app.draw_power_state_graph()

######### TV
tv_app = Appliance('tv')
tv1 = Channel()
tv1.load_wattsup(path.join(APP_DATA_DIR, 'tv1.csv'))
tv_app.train_on_single_example(tv1)

######## Toaster
toaster_app = Appliance('toaster')
t1 = Channel()
t1.load_wattsup(path.join(APP_DATA_DIR, 'toaster1.csv'))
toaster_app.train_on_single_example(t1)

######## Kettle
kettle_app = Appliance('kettle')
k1 = Channel()
k1.load_wattsup(path.join(APP_DATA_DIR, 'kettle1.csv'))
kettle_app.train_on_single_example(k1)

######### TRAIN
full_active = Channel()
full_active.load_normalised(DD, high_freq_param='active')

disag = BayesDisaggregator()
disag.train([wm_app, tv_app, toaster_app, kettle_app], aggregate=full_active)
# disag.train([toaster_app, kettle_app])
ax = plt.gca()
ax.bar(disag._bin_edges[:-1], disag._prob_mass)
plt.show()

######### DISAGGREGATE!!!

predictions, pwr_segs = disag.disaggregate(agg_chan, return_power_segments=True)

plot_appliance_hypotheses(subplots[2], predictions)

print([app.label for app in predictions['appliance']])

plot_steady_states(subplots[0], pwr_segs)

plt.show()
