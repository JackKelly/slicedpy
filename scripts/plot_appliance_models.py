from pda.channel import Channel
from slicedpy.appliance import Appliance
import matplotlib.pyplot as plt

c = Channel()
c.load_wattsup('/data/mine/domesticPowerData/BellendenRd/wattsUp/washingmachine1.csv')

app = Appliance()
sig_power_states = app.train_on_single_example(c)

fig, ax = plt.subplots()
c.plot(ax)
# for ps in sig_power_states:
#    ps.plot(ax)
plt.show()
