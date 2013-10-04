from __future__ import print_function, division
from pda.channel import Channel
from slicedpy.appliance import Appliance
import matplotlib.pyplot as plt
from os import path

DATA_DIR = '/data/mine/domesticPowerData/BellendenRd/wattsUp'

def train_appliance(label, sig_data_filenames):
    """
    Args:
      * label (str): e.g. 'washing machine'
      * sig_data_filenames (list of strings): filenames of signature data files
    """
    app = Appliance(label)
    
    # Train
    for f_name in sig_data_filenames:
        print('Loading', f_name)
        chan = Channel()
        chan.load_wattsup(path.join(DATA_DIR, f_name))
        sps = app.train_on_single_example(chan)

    # Plot raw power data
    fig1, ax1 = plt.subplots()
    chan.plot(ax1, date_format='%H:%M:%S')
    ax1.set_title(f_name)

    # Plot power segments
    for ps in sps: # power segment in signature power segment
        ps.plot(ax1)

    # Draw power state graph
    app.draw_power_state_graph()

    # Print out some useful info
    for node in app.power_state_graph.nodes():
        print(node)
        print("node", node.power.get_model().mean, "essential=", node.essential)


# train_appliance('washing machine', ['washingmachine1.csv', 'washingmachine2.csv'])
train_appliance('tv', ['tv1.csv'])
# train_appliance('toaster', ['toaster1.csv'])
# train_appliance('kettle', ['kettle1.csv'])

plt.show()
