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
import slicedpy.disaggregator as sd
from test_appliance import fake_signature

class TestDisaggregator(unittest.TestCase):

    def test_get_bins(self):
        sig = fake_signature()
        fwd_diff = sig.series.diff().dropna().values
        fwd_diff = fwd_diff[np.fabs(fwd_diff) >= sd.MIN_FWD_DIFF]
        # fwd_diff is now = [100, 100, -100, 100, -150, -50]

        bin_edges, n_negative_bins = sd.get_bins(fwd_diff)

        neg_bins = np.arange(-150 - sd.MIN_FWD_DIFF, 
                             -sd.MIN_FWD_DIFF, 
                             sd.BIN_WIDTH)
        pos_bins = np.arange(sd.MIN_FWD_DIFF, 
                             100 + sd.MIN_FWD_DIFF + sd.BIN_WIDTH, 
                             sd.BIN_WIDTH)
        correct_bin_edges = np.concatenate([neg_bins, [-sd.MIN_FWD_DIFF], pos_bins])
        np.testing.assert_array_equal(bin_edges, correct_bin_edges)
        self.assertEqual(n_negative_bins, 30)

    def test_fit_p_fwd_diff(self):
        sig = fake_signature()
        d = sd.BayesDisaggregator()
        d._fit_p_fwd_diff(sig, plot=False)
        print(d._bin_edges[0], d._bin_edges[1])
        self.assertAlmostEqual(d._prob_mass[-1], 3/6) # bin=96-101
        self.assertAlmostEqual(d._prob_mass[0], 1/6) # bin=-151 to -146
        self.assertAlmostEqual(d._p_fwd_diff(100), 3/6)
        self.assertAlmostEqual(d._p_fwd_diff(-150), 1/6)
        self.assertAlmostEqual(d._p_fwd_diff(-146), 1/12)
        self.assertAlmostEqual(d._p_fwd_diff(-147), (7/10)*(1/6))
        self.assertAlmostEqual(d._p_fwd_diff(0), 0)
        self.assertAlmostEqual(d._p_fwd_diff(-48.5), 1/6)
        self.assertAlmostEqual(d._p_fwd_diff(1000), 0)
        self.assertAlmostEqual(d._p_fwd_diff(-1000), 0)

if __name__ == '__main__':
    unittest.main()
