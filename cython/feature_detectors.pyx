"""
RESOURCES:
http://wesmckinney.com/blog/?p=278
http://docs.cython.org/src/userguide/numpy_tutorial.html
"""

from __future__ import print_function, division
import numpy as np
cimport numpy as np
import pandas as pd
# from slicedpy.feature import Feature

# Data types for timestamps (TS = TimeStamp)
TS_DTYPE = np.uint64
ctypedef np.uint64_t TS_DTYPE_t

# Data types for power data (PW = PoWer)
PW_DTYPE = np.float32
ctypedef np.float32_t PW_DTYPE_t

def steady_state(np.ndarray[object] watts,
                 Py_ssize_t min_n_samples=3, 
                 PW_DTYPE_t max_range=15):
    """Steady_state detector based on the definition of steady states given
    in Hart 1992, page 1882, under the heading 'C. Edge Detection'.

    Args:
        watts (np.ndarray): Watts.
        min_n_samples (int): Optional. Defaults to 3. Minimum number of 
            consecutive samples per steady state.  Hart used 3.
        max_range (float): Optional. Defaults to 15 Watts. Maximum 
            permissible range between the lowest and highest value per
            steady state. Hart used 15.
    
    Returns:
        List of Features.  Each Feature has a 'watts' attribute which gives the
            mean watts for that steady state.
    """
    cdef:
        Py_ssize_t i, n, ss_start_i # steady_state_start_index

    n = len(watts)
    steady_states = []
    ss_start_i = 0

    for i from 1 <= i <= n:
        ss = watts[ss_start_i:i]

        if np.ptp(ss) > max_range: # np.ptp returns the peak-to-peak value
            if (i - ss_start_i) >= min_n_samples:
                # TODO: use Features
                # ss = Feature(start=ss_start_i, end=i, mean=ss.mean())
                steady_states.append((ss_start_i, i, ss.mean()))
            ss_start_i = i
            
    return steady_states
