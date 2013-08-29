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
import slicedpy.utils as utils
import pandas as pd

class TestUtils(unittest.TestCase):
    def test_find_nearest(self):
        N = 24
        idx = pd.date_range('2013-01-01', periods=N, freq='H')
        series = pd.Series(None, index=idx)
        for i in range(N):
            nearest = utils.find_nearest(series, idx[i])
            self.assertEqual(nearest, i)

        idx_ten_mins = pd.date_range('2013-01-01 00:10', periods=N, freq='H')
        for i in range(N):
            nearest = utils.find_nearest(series, idx_ten_mins[i])
            self.assertEqual(nearest, i)

        idx_fifty_mins = pd.date_range('2013-01-01 00:50', periods=N, freq='H')
        for i in range(N-1):
            nearest = utils.find_nearest(series, idx_fifty_mins[i])
            self.assertEqual(nearest, i+1)
        nearest = utils.find_nearest(series, idx_fifty_mins[-1])
        self.assertEqual(nearest, N-1)

        # create events of duration = 50 mins so we can test align='end'
        dicts = []
        for i in range(N):
            dicts.append({'end':idx_fifty_mins[i]})
        df = pd.DataFrame(dicts, index=idx)

        for i in range(N):
            nearest = utils.find_nearest(df, idx_fifty_mins[i], align='end')
            self.assertEqual(nearest, i)

        for i in range(0,N):
            nearest = utils.find_nearest(df, idx[i], align='end')
            self.assertEqual(nearest, i-1 if i > 0 else 0)
            


if __name__ == '__main__':
    unittest.main()
