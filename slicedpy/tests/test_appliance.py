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
from slicedpy.appliance import Appliance
from sklearn.neighbors import NearestNeighbors
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
    def setUp(self):
        # TRAIN POWER STATE GRAPH
        self.chan = fake_signature()
        self.app = Appliance(label='test appliance')
        sig_power_states = self.app.train_on_single_example(self.chan)

        self.nodes = self.app.power_state_graph.nodes()
        self.nodes.sort(key=lambda node: node.power.get_model().mean)

        self.correct_edges = [(self.nodes[0], self.nodes[2]),
                              (self.nodes[2], self.nodes[3]),
                              (self.nodes[3], self.nodes[2]),
                              (self.nodes[3], self.nodes[1]),
                              (self.nodes[1], self.nodes[0])]

        PLOT_DATA = False
        if PLOT_DATA:
            import matplotlib.pyplot as plt
            fig1, ax1 = plt.subplots()
            self.chan.plot(ax1)
            for ps in sig_power_states:
                ps.plot(ax1)

            self.app.draw_power_state_graph()
            plt.show()

    def test_train_on_single_example(self):
        correct_powers = [0, 50, 100, 200]
        correct_sizes = [514, 300, 120, 159]
        correct_edge_pwrs = [100, 100, -100, -150, -50]

        # CHECK SELF.NODES
        self.assertEqual(len(self.nodes), 4)

        for i, (pwr, size) in enumerate(zip(correct_powers, correct_sizes)):
            self.assertEqual(self.nodes[i].power.get_model().mean, pwr)
            self.assertEqual(self.nodes[i].power.get_model().var, 0)
            self.assertEqual(self.nodes[i].power.get_model().size, size)

        # CHECK EDGES
        self.assertEqual(len(self.correct_edges), 
                         len(self.app.power_state_graph.edges()))
        self.assertEqual(set(self.correct_edges), 
                         set(self.app.power_state_graph.edges()))

        for i, edge in enumerate(self.correct_edges):
            e = self.app.power_state_graph[edge[0]][edge[1]]['object']
            edge_pwr = e.power_segment_diff[-1][0]
            self.assertEqual(edge_pwr, correct_edge_pwrs[i])

    def test_edge_knn(self):
        X, Y = self.app.get_edge_feature_matrix()
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(X)
        distances, indices = knn.kneighbors([-50,1])
        self.assertEqual(indices[0][0], 3)

if __name__ == '__main__':
    unittest.main()
