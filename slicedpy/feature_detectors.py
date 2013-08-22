from __future__ import print_function, division
from _cython_feature_detectors import *
import numpy as np
import copy
from scipy import stats
import scipy.optimize
import matplotlib.dates as mdates

"""
This file implements feature detectors which are written in pure
Python.  Cython feature detectors are in
cython/_cython_feature_detectors.pyx.  This file also holds helper functions
for pre-processing prior to using feature detectors.
"""

###############################################################################
# SPIKE HISTOGRAM FUNCTIONS
###############################################################################

def merge_spikes(fdiff):
    """Merge consecutive forward difference values of the same sign.

    Args:
        fdiff (1D np.ndarray): forward difference of power. 
            e.g. calculated by np.diff

    Returns: 
        merged_fdiff (1D np.ndarray).  Will be zero prior to each merged spike.
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


###############################################################################
# multiple_linear_regressions
###############################################################################

def multiple_linear_regressions(data, window_size=10):
    """
    Args:
        data (np.ndarray): power.
        window_size (int): Width of each windows in number of samples.  
            Must be multiple of 2.  Windows are overlapping:
              2222
            11113333
    Return:
        np.ndarray(n_windows, 4): the 4 columns are: 
            slope, intercept, r_value**2, std_err
    """
    assert((window_size % 2) == 0)

    half_window_size = window_size / 2
    n_windows = int(data.size / half_window_size) - 1
    x = np.arange(window_size)
    results = np.empty((n_windows, 4))
    start_i = 0
    end_i = window_size
    for i in range(n_windows):
        window = data[start_i:end_i]
        start_i += half_window_size
        end_i += half_window_size
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, window)
        results[i] = (slope, intercept, r_value**2, std_err)

    return results


###############################################################################
# SPIKE THEN DECAY
###############################################################################

def spike_indices(data, min_spike_size, max_spike_size=None):
    # Find spikes between min_spike_size and max_spike_size
    fdiff = np.diff(data)
    fdiff = merge_spikes(fdiff)
    if max_spike_size is None:
        spike_indices = np.where(fdiff > min_spike_size)[0]
    else:
        assert(max_spike_size > min_spike_size)
        spike_indices = np.where((fdiff > min_spike_size) & 
                                 (fdiff < max_spike_size))[0]
    spike_indices += 1
    return spike_indices


def spike_then_decay(series, min_spike_size=600, max_spike_size=None, 
                            decay_window=10, mode='linear'):

    def linear(f, x, values):
        (f.slope, _, f.r_value, 
         f.p_value, f.stderr) = stats.linregress(x, values)

    curve = lambda x, c, m: c + (m / x)
    def poly(f, x, values):        
        f.popt, f.pconv = scipy.optimize.curve_fit(curve, (x-x[0])+1, values,
                                                   p0=(values.min(), 281.5))

    if mode=='linear':
        regression_func = linear
    elif mode=='poly':
        regression_func = poly
    else:
        raise Exception('Mode \'' + mode + '\' not recognised')

    spike_idxs = spike_indices(series.values[:-decay_window], 
                               min_spike_size, max_spike_size)

    # For each spike, do linear regression of next decay_window values
    features = []
    for spike_i in spike_idxs:
        f = Feature(start=spike_i, end=spike_i+decay_window)
        chunk = series[f.start:f.end]
        x = mdates.date2num(chunk.index) * mdates.SEC_PER_DAY
        regression_func(f, x, chunk.values)
        features.append(f)

    return features

###############################################################################
# POWER STATE DETECTORS
###############################################################################
def relative_deviation_power_states(
                  watts,
                  rdt=0.05, # relative deviation threshold
                  window_size=10):
    """Power state detector designed to find "power states" in the face of
    a rapidly oscillating signal (e.g. a washing machine's motor).

    Break *watts* into chunks, each of size *window_size*.  Calculate the mean
    of the first chunk. Calculate the mean deviation of the second chunk
    against the mean of the first chunk.  If this deviation is above *rdt*
    then we've left a candidate power state.  If we've been through 2 or
    more chunks then store this power state.  Repeat.
    """

    # TODO: Optimise:
    # * convert code to Cython 
    # * don't calculate ps.mean() from scratch every iteration

    def mean_relative_deviation(next_chunk, ps_mean):
        """Convert to absolute value _after_ calculating the mean.
        The idea is that rapid oscillations should cancel themselves out."""
        return np.fabs((next_chunk - ps_mean).mean() / ps_mean)

    n_chunks = int(watts.size / window_size)
    print("n_chunks =", n_chunks)
    ps_start_i = 0
    ps_end_i = window_size
    power_states = []
    for chunk_i in range(n_chunks-2):
        ps = watts[ps_start_i:ps_end_i]
        next_chunk_end_i = (chunk_i+2)*window_size
        next_chunk = watts[ps_end_i:next_chunk_end_i]
        if mean_relative_deviation(next_chunk, ps.mean()) > rdt:
            # new chunk marks the end of the power state
            if (ps_end_i - ps_start_i) / window_size > 1:
                feature = Feature(start=ps_start_i, end=ps_end_i, mean=ps.mean())
                power_states.append(feature)
            ps_start_i = ps_end_i
        ps_end_i = next_chunk_end_i

    return power_states


def min_max_power_states(watts,
                          max_deviation=20, # watts
                          initial_window_size=30, # int: number of samples
                          look_ahead=3, # int: number of samples
                          max_ptp=1000 # max peak to peak in watts
                          ):
    ps_start_i = 0
    ps_end_i = initial_window_size
    power_states = []
    half_window = int(initial_window_size / 2)
    n = watts.size - look_ahead
    while True:
        if ps_end_i >= n:
            break

        ps = watts[ps_start_i:ps_end_i]
        end_of_ps = False
        if ps.size == initial_window_size:
            # this is an initial chunk so test it's a sane
            # chunk by comparing the means of the left and right side
            # of this chunk
            halfway = ps_start_i + half_window
            left = watts[ps_start_i:halfway]
            right = watts[halfway:ps_end_i]
            if (np.fabs(left.min() - right.min()) > max_deviation or
                np.fabs(left.max() - right.max()) > max_deviation or
                ps.ptp() > max_ptp):
                ps_start_i += 1
                ps_end_i += 1
                continue
        else:
            # Take an *initial_window_size* chunk from the tail
            # and make sure the front of the power state
            # falls within the min and max of the tail.
            tail_split = ps_end_i-initial_window_size
            front = watts[ps_start_i:tail_split]
            tail = watts[tail_split:ps_end_i]
            if (front.mean() < tail.min() - max_deviation or
                front.mean() > tail.max() + max_deviation):
                end_of_ps = True
                ps_end_i = tail_split

        ahead = watts[ps_end_i:ps_end_i+look_ahead]
        if (ahead.mean() < ps.min() - max_deviation or
            ahead.mean() > ps.max() + max_deviation or
            end_of_ps):
            # We've come to the end of a candidate power state
            feature = Feature(start=ps_start_i, end=ps_end_i-1,
                              mean=ps.mean())
            power_states.append(feature)
            ps_start_i = ps_end_i
            ps_end_i = ps_start_i + initial_window_size
        else:
            ps_end_i += 1

    return power_states


def min_max_two_halves_power_states(watts, 
                                    max_deviation=20, # watts
                                    initial_window_size=20, # int: n samples
                                    max_ptp=1000 # max peak to peak in watts
                                    ):
    ps_start_i = 0
    ps_end_i = initial_window_size
    power_states = []
    while True:
        if ps_end_i >= watts.size:
            break

        ps = watts[ps_start_i:ps_end_i]

        # test it's a sane
        # chunk by comparing the means of the left and right side
        halfway = int((ps_start_i + ps_end_i) / 2)
        left = watts[ps_start_i:halfway]
        right = watts[halfway:ps_end_i]
        
        if (np.fabs(left.min() - right.min()) > max_deviation or
            np.fabs(left.max() - right.max()) > max_deviation or
            ps.ptp() > max_ptp):
            if ps_end_i == ps_start_i + initial_window_size:
                ps_start_i += 1
                ps_end_i += 1
            else:
                # We've come to the end of a candidate power state
                feature = Feature(start=ps_start_i, end=ps_end_i-1, 
                                  mean=ps.mean())
                power_states.append(feature)
                ps_start_i = ps_end_i
                ps_end_i = ps_start_i + initial_window_size
        else:
            ps_end_i += 1

    return power_states


def mean_chunk_power_states(watts, max_deviation=10, window_size=10):
    ps_start_i = 0
    power_states = []
    n_chunks = int(watts.size / window_size)
    for chunk_i in range(1,n_chunks-2):
        ps_end_i = window_size * chunk_i
        ps = watts[ps_start_i:ps_end_i]
        next_chunk = watts[ps_end_i:ps_end_i+window_size]
        if (next_chunk.mean() > ps.mean() + max_deviation or
            next_chunk.mean() < ps.mean() - max_deviation):
            # We've come to the end of a candidate power state
            feature = Feature(start=ps_start_i, end=ps_end_i, mean=ps.mean())
            power_states.append(feature)
            ps_start_i = ps_end_i

    return power_states


def minimise_mean_deviation_power_states(watts,
                                         max_deviation=20, # watts
                                         initial_window_size=30, # int: number of samples
                                         look_ahead=150, # int: number of samples
                                         max_ptp=1000 # max peak-to-peak watts
                                         ):
    ps_start_i = 0
    ps_end_i = initial_window_size
    power_states = []
    quarter_window = int(initial_window_size / 4)
    half_window = int(initial_window_size / 2)
    n = watts.size - look_ahead
    while True:
        if ps_end_i >= n:
            break

        # start_i = ps_start_i
        # end_i = ps_end_i
        # ps = watts[start_i:end_i]

        # if ps_end_i == ps_start_i + initial_window_size:
        #     # Tweak start and end to minimise skew
        #     min_skew = np.finfo(float).max
        #     start_i = None
        #     end_i = None
        #     for start_i in range(ps_start_i, ps_start_i+quarter_window):
        #         for end_i in range(ps_end_i-quarter_window, ps_end_i):
        #             ps = watts[start_i:end_i]
        #             skew = np.fabs(stats.skew(ps))
        #             if skew < min_skew:
        #                 min_skew = skew
        #                 start_i = start_i
        #                 end_i = end_i
        #     ps_start_i = start_i
        #     ps_end_i = end_i

        # this is an initial chunk so test it's a sane
        # chunk by comparing the means of the left and right side
        # of this chunk
        ps = watts[ps_start_i:ps_end_i]
        if ps_end_i == ps_start_i + initial_window_size:
            halfway = ps_start_i + half_window
            left = watts[ps_start_i:halfway]
            right = watts[halfway:ps_end_i]
            if (np.fabs(left.min() - right.min()) > max_deviation or
                np.fabs(left.max() - right.max()) > max_deviation or
                ps.ptp() > max_ptp):
                ps_start_i += 1
                ps_end_i += 1
                continue

        # Now creep forwards from ps_end_i and find the index which
        # gives the lowest deviation of the mean of the power state
        # against the mean of the look ahead chunk.
        min_deviation = np.finfo(np.float64).max
        i_of_lowest = None
        ps_mean = ps.mean()
        for look_ahead_i in range(ps_end_i+1, ps_end_i+look_ahead):
            ahead = watts[ps_end_i:look_ahead_i]
            # TODO: don't recalculate ahead.mean() from scratch every iteration
            deviation = np.fabs(ahead.mean() - ps_mean)
            if deviation <= min_deviation:
                min_deviation = deviation
                i_of_lowest = look_ahead_i

        # Now we've found the index of the minimum deviation
        if min_deviation <= max_deviation:
            ps_end_i = i_of_lowest
        else:
            # End of power state
            feature = Feature(start=ps_start_i, end=ps_end_i-1, mean=ps.mean())
            power_states.append(feature)
            ps_start_i = ps_end_i
            ps_end_i = ps_start_i + initial_window_size

    return power_states
