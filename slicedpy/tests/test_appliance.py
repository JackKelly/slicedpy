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
import slicedpy.appliance as app
from slicedpy.bunch import Bunch
from slicedpy.powerstate import PowerState
from slicedpy.datastore import DataStore
from slicedpy.normal import Normal

class TestPowerState(unittest.TestCase):
    def test_update_power_state_graph(self):

        powers = [Normal(mean=100, var=10, size=10), 
                  Normal(mean=103, var=10, size=10), 
                  Normal(mean=300, var=10, size=10), 
                  Normal(mean=300, var=10, size=10), 
                  Normal(mean=999, var=99, size=10)]

        sig_pwr_sgmnts = [PowerState(start=  0, end=200, power=DataStore(model=powers[0])),
                          PowerState(start=400, end=500, power=DataStore(model=powers[1])),
                          PowerState(start=501, end=600, power=DataStore(model=powers[2])),
                          PowerState(start=601, end=700, power=DataStore(model=powers[3])),
                          PowerState(start=701, end=800, power=DataStore(model=powers[4]))]

        app.update_power_state_graph(sig_pwr_sgmnts)
        nodes = app.power_state_graph.nodes()

        # Test unique power states
        correct_pwrs = [Normal(var=11.9205298013, size=300, mean=101.),
                        Normal(var= 9.9,          size=198, mean=300.),
                        Normal(var=99.,           size= 99, mean=999.)]

        correct_ups = [PowerState(power=DataStore(model=correct_pwrs[0])),
                       PowerState(power=DataStore(model=correct_pwrs[1])),
                       PowerState(power=DataStore(model=correct_pwrs[2]))]

        for corr_ups, test_ups in zip(correct_ups, nodes):
            self.assertEqual(corr_ups.power.model.size, test_ups.power.model.size)
            self.assertEqual(corr_ups.power.model.mean, test_ups.power.model.mean)
            self.assertAlmostEqual(corr_ups.power.model.var, test_ups.power.model.var)

        # Test mapped signature power segments
        correct_msps = [Bunch(start=  0, end=200, mean=100, var=10, power_state=0),
                        Bunch(start=400, end=500, mean=103, var=10, power_state=0),
                        Bunch(start=501, end=600, mean=300, var=10, power_state=1),
                        Bunch(start=601, end=700, mean=300, var=10, power_state=1),
                        Bunch(start=701, end=800, mean=999, var=99, power_state=2)]
        
        for corr_msps, test_msps in zip(correct_msps, mapped_sig_pwr_sgmnts):
            self.assertEqual(corr_msps, test_msps)


if __name__ == '__main__':
    unittest.main()
