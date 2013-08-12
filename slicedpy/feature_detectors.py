from _cython_feature_detectors import *
import numpy as np

"""
This file implements feature detectors which are written in pure
Python.  Cython feature detectors are in
cython/_cython_feature_detectors.pyx.
"""

def merge_spikes(fdiff):
    """
    Merge consecutive forward difference values of the same sign.

    Args:
        fdiff (1D np.ndarray): forward difference of power. 
            e.g. calculated by np.diff

    Returns: 
        merged_fdiff (1D np.ndarray).  Will be zero where 
    """
    import ipdb; ipdb.set_trace()
    sign_comparison = (fdiff[:-1] * fdiff[1:]) > 0
    merged_fdiff = np.zeros(sign_comparison.size)
    accumulator = 0
    for i in range(1,sign_comparison.size-1):
        if sign_comparison[i] == True:
            if accumulator == 0:
                accumulator = fdiff[i] + fdiff[i+1]
            else:
                accumulator += fdiff[i+1]
        else:
            if accumulator == 0:
                merged_fdiff[i] = fdiff[i]
            else:
                merged_fdiff[i] = accumulator
                accumulator = 0

    return merged_fdiff
