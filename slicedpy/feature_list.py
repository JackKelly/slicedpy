from __future__ import division, print_function
import numpy as np

class FeatureList(list):
    """Container for :class:`Feature`s (or any class with ``start`` and ``end``)
    """

    def find_nearest(self, target_ts_index, align='start'):
        """
        Args:
          * target_ts_index (int): the series timestamp index for which 
            to find a nearby Features.
          * align (str): 'start' | 'end`

        Returns:
          index into list of item with nearest ``start`` or ``end`` 
          to target_ts_index.
          If FeatureList is empty then return None.
          If FeatureList has only one item then 0.

        Raises:
          AttributeError if we encounter a list item without a 
            ``start`` or ``end``.
        """

        assert(align in ['start', 'end'])
        if target_ts_index < 0:
            raise IndexError('target_ts_index must be positive.')

        list_length = len(self)
        if list_length == 0:
            return None
        elif list_length == 1:
            return 0

        get_start_ts_index = lambda i: self[i].start
        get_end_ts_index = lambda i: self[i].end
        get_ts_index = get_start_ts_index if align=='start' else get_end_ts_index

        best_list_index = int(round((target_ts_index / get_ts_index(-1)) * 
                                     list_length))

        if best_list_index >= list_length:
            best_list_index = list_length - 1

        get_diff = lambda i: abs(get_ts_index(i) - target_ts_index)

        min_diff = get_diff(best_list_index)
        # Search forwards
        for i in range(best_list_index+1, list_length):
            diff = get_diff(i)
            if diff < min_diff:
                min_diff = diff
                best_list_index = i
            else: # diff is rising                
                break
            
        # Search backwards
        for i in range(best_list_index-1, -1, -1):
            diff = get_diff(i)
            if diff < min_diff:
                min_diff = diff
                best_list_index = i
            else: # diff is rising                
                break

        return best_list_index
