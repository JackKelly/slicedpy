from _cython_feature_detectors import *
import numpy as np
import copy

"""
This file implements feature detectors which are written in pure
Python.  Cython feature detectors are in
cython/_cython_feature_detectors.pyx.  This file also holds helper functions
for pre-processing prior to using feature detectors.
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

    sign_comparison = (fdiff[:-1] * fdiff[1:]) > 0
    merged_fdiff = copy.copy(fdiff)
    accumulator = 0
    for i in range(0,merged_fdiff.size-1):
        if sign_comparison[i] == True:
            if accumulator == 0:
                accumulator = fdiff[i] + fdiff[i+1]
            else:
                accumulator += fdiff[i+1]
            merged_fdiff[i] = 0
        else:
            if accumulator != 0:
                merged_fdiff[i] = accumulator
                accumulator = 0

    # Handle last element if necessary
    if accumulator != 0:
        merged_fdiff[-1] = accumulator

    return merged_fdiff


def spike_histogram(fdiff, window_size=60, n_bins=8):
    """
    Args:
        fdiff (np.array): forward difference of power signal calculated by,
            for example, np.diff().  You may want to merge spikes using
            merge_spikes() before passing fdiff to spike_histogram().
        window_size (int): number of samples per window.
        n_bins (int): number of bins per window.
    """
    abs_fdiff = np.fabs(fdiff)
    n_chunks = int(abs_fdiff.size / window_size)
    start_i = 0
    bin_edges = np.concatenate(([0], np.exp(np.arange(1,n_bins+1))))
    print("bin edges =", bin_edges)
    print("n_bins=", n_bins, "n_chunks=", n_chunks)
    spike_hist = np.empty((n_bins, n_chunks), dtype=np.int64)
    for chunk_i in range(n_chunks):
        end_i = (chunk_i+1)*window_size
        chunk = abs_fdiff[start_i:end_i]
        spike_hist[:,chunk_i] = np.histogram(chunk, bins=bin_edges)[0]
        start_i = end_i
    return spike_hist, bin_edges


def spike_histogram_row_to_data_coordinates(row):
    """make a 2d matrix X where each row stores the coordinates 
    of a single data point."""
    nonzero_i = np.nonzero(row)[0]
    X = np.zeros((nonzero_i.size, 2), dtype=np.float64)
    X[:,0] = nonzero_i
    X[:,1] = row[nonzero_i]
    return X
