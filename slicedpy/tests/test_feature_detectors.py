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
import slicedpy.feature_detectors as fd
import numpy as np

class TestFeatureDetectors(unittest.TestCase):
    def test_steady_states(self):
        arr = np.array([1,2,3,4,5,6,100,101,102,103,104,115, 116], 
                       dtype=np.float32)
        
        steady_states = fd.steady_states(arr)
        self.assertEqual(steady_states[0], (0, 6,3.5))
        self.assertEqual(steady_states[1][0], 6)
        self.assertEqual(steady_states[1][1], 12)
        self.assertAlmostEqual(steady_states[1][2], 104.166, places=2)

        #########################
        # Now try to break it...
    
        # Wrong dtype
        arr_wrong_dtype = np.array([1,2,3,4,5,6], dtype=long)
        with self.assertRaises(ValueError):
            fd.steady_states(arr_wrong_dtype)

        # Wrong number of dimensions
        arr_wrong_ndim = np.array([[1,2,3,4,5,6],[1,2,3,4,5,6]], dtype=np.float32)
        with self.assertRaises(ValueError):
            fd.steady_states(arr_wrong_ndim)

        # Too few elements
        arr_too_small = np.array([1,2], dtype=np.float32)
        with self.assertRaises(ValueError):
            fd.steady_states(arr_too_small)

        ##########################################
        # Now do a quick bit of basic profiling...


if __name__ == '__main__':
    unittest.main()
