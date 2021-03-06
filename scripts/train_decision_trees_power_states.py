from pda.channel import Channel
from slicedpy.appliance import Appliance
from slicedpy.disaggregator import DTDisaggregator
import matplotlib.pyplot as plt
import os.path as path

DATA_DIR = '/data/mine/domesticPowerData/BellendenRd/wattsUp'

######### WASHING MACHINE
wm_app = Appliance('wm')

wm1 = Channel()
wm1.load_wattsup(path.join(DATA_DIR, 'washingmachine1.csv'))
wm_app.train_on_single_example(wm1)

wm2 = Channel()
wm2.load_wattsup(path.join(DATA_DIR, 'washingmachine2.csv'))
wm_sig_power_states = wm_app.train_on_single_example(wm2)

fig1, ax1 = plt.subplots()
wm2.plot(ax1)
for ps in wm_sig_power_states:
    ps.plot(ax1)

wm_app.draw_power_state_graph()

######### TV
tv_app = Appliance('tv')
tv1 = Channel()
tv1.load_wattsup(path.join(DATA_DIR, 'tv1.csv'))
tv_app.train_on_single_example(tv1)

######## Toaster
toaster_app = Appliance('toaster')
t1 = Channel()
t1.load_wattsup(path.join(DATA_DIR, 'toaster1.csv'))
toaster_app.train_on_single_example(t1)

######## Kettle
kettle_app = Appliance('kettle')
k1 = Channel()
k1.load_wattsup(path.join(DATA_DIR, 'kettle1.csv'))
kettle_app.train_on_single_example(k1)

######### TRAIN DECISION TREE
disag = DTDisaggregator()
disag.train_decision_tree([wm_app, tv_app, toaster_app, kettle_app])
disag.draw_tree()

plt.show()
