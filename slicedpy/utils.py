from __future__ import division, print_function
import pandas as pd
import numpy as np

def find_nearest(data, target, align='start', max_time_diff=None):
    """Finds the index of the nearest row in `data` to `target` time.

    Args:
      * data (pd.Series or pd.DataFrame): if `align==back` then `data` must
        have an `end` column.
      * target (pd.Timeseries or timestamp)
      * align (str): `start` or `end`.  Align with the front of the event
        (as recorded by `data.index`) or the back (as recorded by `data['end']`)
      * max_time_diff (datetime.timedelta or None): optional.

    Returns:
      int. Index into `data` for the element nearest in time to `target`.
    
    """

    assert(align in ['start','end'])

    if isinstance(target, pd.Timestamp):
        target = target.to_pydatetime()

    if align == 'start':
        diff = data.index.to_pydatetime() - target
    else:
        diff = pd.to_datetime(data['end']) - target

    diff = np.abs(diff)
    min_diff = diff.min()
    
    if max_time_diff is not None and min_diff > max_time_diff:
        return None
    else:
        return diff.argmin()
    
