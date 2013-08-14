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
from slicedpy.feature import Feature
import numpy as np
import pandas as pd
import scipy.stats as stats

class TestFeature(unittest.TestCase):
    def test_ttest_both_halves(self):
        N = 5000
        rvs = stats.norm.rvs(loc=100, scale=20, size=N)
        f = Feature(0, N)
        p_value = f.ttest_both_halves(rvs)
        self.assertEqual(f.p_value_both_halves, p_value)

    def test_linregress(self):
        N = 20
        idx = pd.date_range(start='2012/1/1', freq='S', periods=N)

        # Horizontal line...
        d = np.zeros(N) * 10
        s = pd.Series(d, idx)
        f = Feature(0, N)
        f.linregress(s)
        self.assertEqual(f.slope, 0)

        # Line with gradient slope 1
        d = np.linspace(1, N, N)
        s = pd.Series(d, idx)
        f = Feature(0, N)
        f.linregress(s)
        print('{:.20f}'.format(f.slope))
        self.assertAlmostEqual(f.slope, 1, places=5)

if __name__ == '__main__':
    unittest.main()
