"""
RESOURCES:
http://wesmckinney.com/blog/?p=278
http://docs.cython.org/src/userguide/numpy_tutorial.html
"""

from __future__ import print_function, division
import numpy as np
cimport numpy as np
cimport cython
import pandas as pd
# from slicedpy.feature import Feature

# Data types for timestamps (TS = TimeStamp)
TS_DTYPE = np.uint64
ctypedef np.uint64_t TS_DTYPE_t

# Data types for power data (PW = PoWer)
PW_DTYPE = np.float32
ctypedef np.float32_t PW_DTYPE_t

# @cython.boundscheck(False)  # turn off bounds-checking for entire function
def steady_states(np.ndarray[PW_DTYPE_t, ndim=1] watts,
                 Py_ssize_t min_n_samples=3, 
                 PW_DTYPE_t max_range=15):
    """Steady_state detector based on the definition of steady states given
    in Hart 1992, page 1882, under the heading 'C. Edge Detection'.

    Args:
        watts (np.ndarray): Watts. Row vector. np.float_32
        min_n_samples (int): Optional. Defaults to 3. Minimum number of 
            consecutive samples per steady state.  Hart used 3.
        max_range (float): Optional. Defaults to 15 Watts. Maximum 
            permissible range between the lowest and highest value per
            steady state. Hart used 15.
    
    Returns:
        List of Features.  Each Feature has a 'watts' attribute which gives the
            mean watts for that steady state.
    """

    # Sanity checking
    if watts is None or min_n_samples is None or max_range is None:
        raise ValueError('Do not use None for any arguments.')
    if watts.size < min_n_samples:
        raise ValueError('watts array must have more than '
                         'min_n_samples={} elements!'.format(min_n_samples))

    cdef:
        Py_ssize_t i, n, ss_start_i # steady_state_start_index
        PW_DTYPE_t p, ss_max, ss_min # steady state max and mins

    n = len(watts)
    ss = [] # steady states. What we return
    ss_start_i = 0
    ss_min = ss_max = watts[ss_start_i]

    for i from 1 <= i < n:
        p = watts[i]

        if p > ss_max:
            ss_max = p
        if p < ss_min:
            ss_min = p

        if (ss_max - ss_min) > max_range: # Just left a candidate steady state.
            if (i - ss_start_i) >= min_n_samples:
                # TODO: use Features
                # ss = Feature(start=ss_start_i, end=i, mean=ss.mean())
                ss.append((ss_start_i, i, watts[ss_start_i:i].mean()))
            ss_start_i = i
            ss_min = ss_max = watts[ss_start_i]

    if (i - ss_start_i) >= min_n_samples:
        # TODO: use Features
        # ss = Feature(start=ss_start_i, end=i, mean=ss.mean())
        ss.append((ss_start_i, i, watts[ss_start_i:i].mean()))
            
    return ss
