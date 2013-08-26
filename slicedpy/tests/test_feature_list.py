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
from slicedpy.powersegment import PowerSegment
from slicedpy.feature_list import FeatureList

class TestFeatureList(unittest.TestCase):
    def test_find_nearest(self):
        # Test with no items
        fl = FeatureList([])
        nearest = fl.find_nearest(100)
        self.assertEqual(nearest, None)

        # Test with 1 item
        fl = FeatureList([PowerSegment(start=  0, end=200, mean=100, var=10)])
        nearest = fl.find_nearest(100)
        self.assertEqual(nearest, 0)

        # Test with several PowerSegments
        pwr_sgmnts = [PowerSegment(start=  0, end=200, mean=100, var=10),
                      PowerSegment(start=400, end=500, mean=103, var=10),
                      PowerSegment(start=501, end=600, mean=300, var=10),
                      PowerSegment(start=601, end=700, mean=300, var=10),
                      PowerSegment(start=701, end=800, mean=999, var=99)]
        fl = FeatureList(pwr_sgmnts)
        self.assertEqual(fl.find_nearest(100), 0)
        self.assertEqual(fl.find_nearest(800), 4)
        self.assertEqual(fl.find_nearest(490), 2)
        self.assertEqual(fl.find_nearest(490, align='end'), 1)
        self.assertEqual(fl.find_nearest(99999999), 4)
        self.assertRaises(IndexError, fl.find_nearest, -10)

        # Test with items which don't contain start or end
        fl = FeatureList([1,2,3])
        self.assertRaises(AttributeError, fl.find_nearest, 100)

if __name__ == '__main__':
    unittest.main()
