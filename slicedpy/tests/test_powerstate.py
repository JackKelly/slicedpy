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
import slicedpy.powerstate as ps
from slicedpy.powersegment import PowerSegment
from slicedpy.powerstate import PowerState

class TestPowerState(unittest.TestCase):
    def test_merge_pwr_sgmnts(self):
        sig_pwr_sgmnts = [PowerSegment(start=  0, end=200, mean=100, var=10),
                          PowerSegment(start=400, end=500, mean=103, var=10),
                          PowerSegment(start=501, end=600, mean=300, var=10),
                          PowerSegment(start=601, end=700, mean=300, var=10),
                          PowerSegment(start=701, end=800, mean=999, var=99)]
        (unique_pwr_states, 
         mapped_sig_pwr_sgmnts) = ps.merge_pwr_sgmnts(sig_pwr_sgmnts)

        # Test unique power states
        correct_ups = [PowerState(var=11.9205298013, size=300, mean=101.),
                       PowerState(var= 9.9,          size=198, mean=300.),
                       PowerState(var=99.,           size= 99, mean=999.)]

        for corr_ups, test_ups in zip(correct_ups, unique_pwr_states):
            self.assertEqual(corr_ups.size, test_ups.size)
            self.assertEqual(corr_ups.mean, test_ups.mean)
            self.assertAlmostEqual(corr_ups.var, test_ups.var)

        # Test mapped signature power segments
        correct_msps = [PowerSegment(start=  0, end=200, mean=100, var=10, power_state=0),
                        PowerSegment(start=400, end=500, mean=103, var=10, power_state=0),
                        PowerSegment(start=501, end=600, mean=300, var=10, power_state=1),
                        PowerSegment(start=601, end=700, mean=300, var=10, power_state=1),
                        PowerSegment(start=701, end=800, mean=999, var=99, power_state=2)]
        
        for corr_msps, test_msps in zip(correct_msps, mapped_sig_pwr_sgmnts):
            self.assertEqual(corr_msps, test_msps)


if __name__ == '__main__':
    unittest.main()
