#!/usr/bin/python

"""
   Copyright 2013 Jack Kelly (aka Daniel)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import print_function, division
import unittest
import numpy as np
import pandas as pd
from pda.channel import Channel
import matplotlib.pyplot as plt
from slicedpy.appliance import Appliance
from slicedpy.bunch import Bunch
from slicedpy.powerstate import PowerState
from slicedpy.datastore import DataStore
from slicedpy.normal import Normal

def fake_signature():
    d = [0]*60 + [100]*60 + [200]*60 + [100]*60 + [200]*100 + [50]*300 + [0]*60
    watts = np.array(d, dtype=np.float32)
    rng = pd.date_range(0, freq='S', periods=len(d))
    series = pd.Series(watts, index=rng)
    return Channel(series=series)

class TestAppliance(unittest.TestCase):
    def test_train_on_single_example(self):
        # TRAIN POWER STATE GRAPH
        chan = fake_signature()
        app = Appliance(label='test appliance')
        sig_power_states = app.train_on_single_example(chan)

        # CHECK NODES
        nodes = app.power_state_graph.nodes()
        nodes.sort(key=lambda node: node.power.get_model().mean)
        self.assertEqual(len(nodes), 4)

        correct_powers = [0, 50, 100, 200]
        correct_sizes = [514, 300, 120, 159]

        for i, (pwr, size) in enumerate(zip(correct_powers, correct_sizes)):
            self.assertEqual(nodes[i].power.get_model().mean, pwr)
            self.assertEqual(nodes[i].power.get_model().var, 0)
            self.assertEqual(nodes[i].power.get_model().size, size)

        # CHECK EDGES
        correct_edges = [(nodes[0], nodes[2]),
                         (nodes[2], nodes[3]),
                         (nodes[3], nodes[2]),
                         (nodes[3], nodes[1]),
                         (nodes[1], nodes[0])]
        self.assertEqual(len(correct_edges), len(app.power_state_graph.edges()))
        self.assertEqual(set(correct_edges), set(app.power_state_graph.edges()))

        # fig1, ax1 = plt.subplots()
        # chan.plot(ax1)
        # for ps in sig_power_states:
        #     ps.plot(ax1)

        # app.draw_power_state_graph()
        # plt.show()

        correct_edge_pwrs = [100, 100, -100, -150, -50]

        for i, edge in enumerate(correct_edges):
            e = app.power_state_graph[edge[0]][edge[1]]['object']
            edge_pwr = e.power_segment_diff.data[-1][0]
            self.assertEqual(edge_pwr, correct_edge_pwrs[i])

if __name__ == '__main__':
    unittest.main()
