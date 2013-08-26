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
from slicedpy.normal import Normal
import scipy.stats as stats
from slicedpy.bunch import Bunch
import numpy as np

def two_samples_from_same_distribtion():
    # stats.norm.rvs(loc=5,scale=10,size=10)
    rvs1 = np.array([  9.08307391, -12.35376395,  13.62230334,  12.69985739,
                       2.5930006 ,  -8.23576804,  10.41061234,  -4.55903562,
                       3.64210533,   0.71815885])
    rvs2 = np.array([ 10.27418946, -20.32813365,  13.52305904,   6.43101003,
                      7.16501859, -14.30066279,   2.54431141,  -6.58077525,
                      4.4993153 ,   5.25385668])
    return rvs1, rvs2

def two_samples_from_different_distribtions():
    # stats.norm.rvs(loc=5,scale=10,size=10)
    rvs1 = np.array([  9.08307391, -12.35376395,  13.62230334,  12.69985739,
                       2.5930006 ,  -8.23576804,  10.41061234,  -4.55903562,
                       3.64210533,   0.71815885])
    # stats.norm.rvs(loc=10,scale=5,size=20)    
    rvs2 = np.array([  3.0752627 ,   9.51583879,  13.34178104,  13.90147932,
                       8.16833051,   0.73092853,   5.21165339,   2.94875162,
                      19.46355455,  11.47967429,  16.44030006,  11.13107523,
                      19.42011661,  15.49147991,   9.33630091,  11.56275982,
                      21.78256937,  15.31409412,   9.22848329,   2.44450088])
    return rvs1, rvs2

class TestStats(unittest.TestCase):

    def test_welch_ttest(self):
        def _test_welch_ttest(rvs1, rvs2):
            norm1 = Normal(values=rvs1)
            norm2 = Normal(values=rvs2)
            wtt = norm1.welch_ttest(norm2)
            stt = stats.ttest_ind(rvs1, rvs2, equal_var=False)[1]
            self.assertAlmostEqual(wtt, stt)

        rvs1, rvs2 = two_samples_from_same_distribtion()
        _test_welch_ttest(rvs1, rvs2)
        rvs1, rvs2 = two_samples_from_different_distribtions()
        _test_welch_ttest(rvs1, rvs2)

    def test_similar_mean(self):
        rvs1, rvs2 = two_samples_from_same_distribtion()
        norm1 = Normal(values=rvs1)
        norm2 = Normal(values=rvs2)
        self.assertTrue(norm1.similar_mean(norm2))

        rvs1, rvs2 = two_samples_from_different_distribtions()
        norm1 = Normal(values=rvs1)
        norm2 = Normal(values=rvs2)
        self.assertFalse(norm1.similar_mean(norm2))

    def test_rough_mean_of_two_normals(self):
        def _test_mean_of_two_normals(rvs1, rvs2):
            rvs3 = np.concatenate([rvs1, rvs2])
            norm1 = Normal(values=rvs1)
            norm2 = Normal(values=rvs2)
            norm3 = norm1.rough_mean_of_two_normals(norm2)
            self.assertEqual(norm3.size, rvs3.size)
            self.assertAlmostEqual(norm3.mean, rvs3.mean())
            self.assertEqual(round(norm3.var), round(rvs3.var()))

        rvs1, rvs2 = two_samples_from_same_distribtion()
        _test_mean_of_two_normals(rvs1, rvs2)
        rvs1, rvs2 = two_samples_from_different_distribtions()
        _test_mean_of_two_normals(rvs1, rvs2)

if __name__ == '__main__':
    unittest.main()
