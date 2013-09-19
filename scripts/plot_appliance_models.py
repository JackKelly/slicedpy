from pda.channel import Channel
from slicedpy.appliance import Appliance
import matplotlib.pyplot as plt

c = Channel()
c.load_wattsup('/data/mine/domesticPowerData/BellendenRd/wattsUp/washingmachine1.csv')

app = Appliance()
sig_power_states = app.train_on_single_example(c)

fig1, ax1 = plt.subplots()
c.plot(ax1)
for ps in sig_power_states:
    ps.plot(ax1)

app.draw_power_state_graph()
plt.show()
