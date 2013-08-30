from __future__ import print_function, division
from _cython_feature_detectors import *
import numpy as np
import pandas as pd
import copy
from scipy import stats
import scipy.optimize
import matplotlib.dates as mdates
import math, datetime
from slicedpy.normal import Normal
from slicedpy.powerstate import PowerState
from slicedpy import utils
from pda.channel import _indicies_of_periods

"""
.. module:: feature_detectors
   :synopsis: Functions for detecting features in power data.
 
   This file implements feature detectors which are written in pure
   Python.  Cython feature detectors are in
   cython/_cython_feature_detectors.pyx.  This file also holds helper functions
   for pre-processing prior to using feature detectors.
"""

###############################################################################
# SPIKE HISTOGRAM FUNCTIONS
###############################################################################

def get_merged_spikes(fdiff):
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


def get_merged_spikes_pandas(series):
    return pd.Series(get_merged_spikes(series.values), index=series.index)


def spike_histogram(series, merge_spikes=True, window_duration=60, n_bins=8):
    """
    Args:
        * series (pd.Series): watts
        * merge_spikes (bool): Default = True
        * window_duration (float): Width of each window in seconds
        * n_bins (int): number of bins per window.

    Returns:
        spike_hist, bin_edges:
            spike_hist (pd.DataFrame):
                index is pd.DateTimeIndex of start of each time window
                columns are 2-tuples of the bin edges in watts (int)
            bin_edges (list of ints):
    """
    fdiff = series.diff()
    if merge_spikes:
        fdiff = get_merged_spikes_pandas(fdiff)        

    abs_fdiff = np.fabs(fdiff)
    freq = (window_duration, 'S')
    date_range, boundaries = _indicies_of_periods(fdiff.index, 
                                                  freq=freq)
    bin_edges = np.concatenate(([0], np.exp(np.arange(1,n_bins+1))))
    bin_edges = np.round(bin_edges).astype(int)
    cols = zip(bin_edges[:-1], bin_edges[1:])
    spike_hist = pd.DataFrame(index=date_range, columns=cols)

    for date_i, date in enumerate(date_range):
        start_i, end_i = boundaries[date_i]
        chunk = abs_fdiff[start_i:end_i]
        spike_hist.loc[date] = np.histogram(chunk, bins=bin_edges)[0]

    return spike_hist, bin_edges


def spike_histogram_bin_to_data_coordinates(bin_data, scale_x=1):
    """make a 2d matrix suitable for clustering where each row stores the
    coordinates of a single data point.  Each x coordinate is the
    ordinal timestamp (in ordinal time where 1 == 1st Jan 0001; 2 =
    2nd Jan 0001 etc), each y coordinate is the count.  Elements with
    count == 0 are ignored.

    Args:
      * bin_data (pd.Series): one bin from the spike histogram.
      * scale_x (float): multiple ordinal time by this value.

    Returns:
      * X (np.ndarray, dim=2, np.float64)

    """
    nonzero_i = np.nonzero(bin_data)[0]
    X = np.empty((nonzero_i.size, 2), dtype=np.float64)
    X[:,0] = mdates.date2num(bin_data[nonzero_i].index) * scale_x
    X[:,1] = bin_data[nonzero_i].values
    return X


###############################################################################
# multiple_linear_regressions
###############################################################################

def multiple_linear_regressions(series, window_size=10):
    """
    Args:
        series (pd.Series): power.
        window_size (int): Width of each windows in number of samples.  
            Must be multiple of 2.  Windows are overlapping:

    ::

                2222
              11113333
    Return:
        np.ndarray(n_windows, 4): the 4 columns are: 
          1. slope
          2. intercept
          3. :math:`\\text{r_value} ^2`
          4. std_err
    """
    assert((window_size % 2) == 0)
    data = series.values
    half_window_size = int(window_size / 2)
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

    idx_i = range(0, half_window_size*n_windows, half_window_size)
    idx = series.index[idx_i]
    return pd.DataFrame(results, index=idx, 
                        columns=['slope','intercept','r_squared','std_err'])


###############################################################################
# DECAYS
###############################################################################

def ttest_both_halves(watts, start, end):
    """
    Test if the left and right half of the watts masked `start` and `end`
    to see if they have the same mean or not.

    Returns two-tailed p-value.

    Args:
      * watts (:class:`pandas.Series`)
      * start (:class:`pandas.Timestamp`)
      * end (:class:`pandas.Timestamp`)
    """
    width = end - start
    half_way = start + width.__div__(2)
    left = watts[(watts.index >= start) & (watts.index < half_way)]
    right = watts[(watts.index >= half_way) & (watts.index < end)]
    return stats.ttest_ind(left, right)[1]

def linregress(watts, start, end):
    """
    Linear regression of data masked `start` and `end`

    Args
      * watts (:class:`pandas.Series`)
      * start (:class:`pandas.Timestamp`)
      * end (:class:`pandas.Timestamp`)

    Returns:
      * ``slope`` (in units of watts per second)
      * ``r_value``
      * ``p_value``
      * ``stderr``
    """
    ss = watts[(watts.index >= start) & (watts.index < end)]
    x = mdates.date2num(ss.index) * mdates.SEC_PER_DAY
    (slope, _, r_value, 
     p_value, stderr) = stats.linregress(x, ss.values)
    return slope, r_value, p_value, stderr


###############################################################################
# SPIKE THEN DECAY
###############################################################################

def spike_indices(data, min_spike_size, max_spike_size=None):
    """Find spikes between min_spike_size and max_spike_size."""
    fdiff = np.diff(data)
    fdiff = get_merged_spikes(fdiff)
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
    """
    Args:
      * series (pd.Series): watts
      * min_spike_size (float): watts
      * max_spike_size (float or None): optional.  watts.
      * decay_window (float): seconds
      * mode (str): 'linear' or 'poly'

    Returns:
      * pd.DataFrame with 'end' and 'decay' columns set.
    """
    def linear(x, values):
        (slope, intercept, r_value, 
         p_value, stderr) = stats.linregress(x, values)
        return intercept, slope

    curve = lambda x, c, m: c + (m / x)
    def poly(x, values):        
        popt, pconv = scipy.optimize.curve_fit(curve, (x-x[0])+1, values,
                                               p0=(values.min(), 281.5))
        return popt[0], popt[1] # intercept, slope

    if mode=='linear':
        regression_func = linear
    elif mode=='poly':
        regression_func = poly
    else:
        raise Exception('Mode \'' + mode + '\' not recognised')

    spike_idxs = spike_indices(series.values[:-decay_window], 
                               min_spike_size, max_spike_size)

    timedelta = datetime.timedelta(seconds=decay_window)

    # For each spike, do regression of next decay_window values
    features = []
    for spike_i in spike_idxs:
        start = series.index[spike_i]
        end = start + timedelta
        chunk = series[spike_i:]
        chunk = chunk[chunk.index < end]
        x = mdates.date2num(chunk.index) * mdates.SEC_PER_DAY
        intercept, slope = regression_func(x, chunk.values)
        f = {'end':end, 'slope':slope, 'intercept':intercept}
        features.append(f)

    return pd.DataFrame(features, index=series.index[spike_idxs])


#############################################################################
# POWER SEGMENT DETECTORS
#############################################################################

def relative_deviation_power_sgmnts(
                  series,
                  rdt=0.05, # relative deviation threshold
                  window_size=10):
    """power segment detector designed to find "power segments" in the face of
    a rapidly oscillating signal (e.g. a washing machine's motor).

    Break ``series`` into chunks, each of size ``window_size``.
    Calculate the mean of the first chunk. Calculate the mean
    deviation of the second chunk against the mean of the first chunk.
    If this deviation is above ``rdt`` then we've left a candidate power
    segment.  If we've been through 2 or more chunks then store this
    power segment.  Repeat.

    """

    # TODO: Optimise:
    # * convert code to Cython 
    # * don't calculate ps.mean() from scratch every iteration

    watts = series.values

    def mean_relative_deviation(next_chunk, ps_mean):
        """Convert to absolute value _after_ calculating the mean.
        The idea is that rapid oscillations should cancel themselves out."""
        return math.fabs((next_chunk - ps_mean).mean() / ps_mean)

    n_chunks = int(watts.size / window_size)
    print("n_chunks =", n_chunks)
    ps_start_i = 0
    ps_end_i = window_size
    idx = [] # index for dataframe
    power_sgmnts = []
    for chunk_i in range(n_chunks-2):
        ps = watts[ps_start_i:ps_end_i]
        next_chunk_end_i = (chunk_i+2)*window_size
        next_chunk = watts[ps_end_i:next_chunk_end_i]
        if mean_relative_deviation(next_chunk, ps.mean()) > rdt:
            # new chunk marks the end of the power segment
            if (ps_end_i - ps_start_i) / window_size > 1:
                idx.append(series.index[ps_start_i])
                power_sgmnts.append({'end': series.index[ps_end_i-1],
                                     'power_stats': Normal(ps)})
            ps_start_i = ps_end_i
        ps_end_i = next_chunk_end_i

    return pd.DataFrame(power_sgmnts, index=idx)


def min_max_power_sgmnts(series, max_deviation=20, initial_window_size=30,
                         look_ahead=3, max_ptp=1000):
    """power segment detector which looks for periods with similar min
    and max values.

    Find "power segments" by first taking an initial window of size
    ``initial_window_size``.  Check that this window is "sane" by
    splitting it in half and comparing the min and max of the two
    halves.  If this window is sane then take the average of
    ``look_ahead`` samples ahead of the end of the window and see if
    this average is within ``max_deviation`` of the min and max of the
    window.  If it is then add ``look_ahead`` onto the window and
    repeat.  Also take the tail of the window and check that the mean
    of the front of the window falls within ``max_deviation`` of the
    min and max of the tail (this helps to make sure we end the power
    segment early enough in a situation where we go from a section with a 
    large min and max to a section with a similar mean but smaller min
    and max).

    Args:
      * series (pd.Series): watts
      * max_deviation (float): watts
      * initial_window_size (int): number of samples
      * look_ahead (int): number of samples
      * max_ptp (float): max peak to peak in watts
    """
    watts = series.values

    ps_start_i = 0
    ps_end_i = initial_window_size
    idx = [] # index for dataframe
    power_sgmnts = []
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
            if (math.fabs(left.min() - right.min()) > max_deviation or
                math.fabs(left.max() - right.max()) > max_deviation or
                ps.ptp() > max_ptp):
                ps_start_i += 1
                ps_end_i += 1
                continue
        else:
            # Take an ``initial_window_size`` chunk from the tail
            # and make sure the mean of the front of the power segment
            # falls within max_deviation of the min and max of the tail.
            tail_split = ps_end_i-initial_window_size
            front = watts[ps_start_i:tail_split]
            tail = watts[tail_split:ps_end_i]
            if (front.mean() < tail.min() - max_deviation or
                front.mean() > tail.max() + max_deviation):
                end_of_ps = True
                ps_end_i = tail_split
                ps = front

        ahead = watts[ps_end_i:ps_end_i+look_ahead]
        if (ahead.mean() < ps.min() - max_deviation or
            ahead.mean() > ps.max() + max_deviation or
            end_of_ps):
            # We've come to the end of a candidate power segment
            idx.append(series.index[ps_start_i])
            power_sgmnts.append({'end': series.index[ps_end_i-1], 
                                 'power_stats': Normal(ps)})
            ps_start_i = ps_end_i
            ps_end_i = ps_start_i + initial_window_size
        else:
            ps_end_i += look_ahead

    return pd.DataFrame(power_sgmnts, index=idx)


def min_max_two_halves_power_sgmnts(series, 
                                    max_deviation=20, # watts
                                    initial_window_size=20, # int: n samples
                                    max_ptp=1000 # max peak to peak in watts
                                    ):
    watts = series.values
    ps_start_i = 0
    ps_end_i = initial_window_size
    idx = [] # index for dataframe
    power_sgmnts = []
    while True:
        if ps_end_i >= watts.size:
            break

        ps = watts[ps_start_i:ps_end_i]

        # test it's a sane
        # chunk by comparing the means of the left and right side
        halfway = int((ps_start_i + ps_end_i) / 2)
        left = watts[ps_start_i:halfway]
        right = watts[halfway:ps_end_i]
        
        if (math.fabs(left.min() - right.min()) > max_deviation or
            math.fabs(left.max() - right.max()) > max_deviation or
            ps.ptp() > max_ptp):
            if ps_end_i == ps_start_i + initial_window_size:
                ps_start_i += 1
                ps_end_i += 1
            else:
                # We've come to the end of a candidate power segment
                idx.append(series.index[ps_start_i])
                power_sgmnts.append({'end': series.index[ps_end_i-1], 
                                     'power_stats': Normal(ps)})
                ps_start_i = ps_end_i
                ps_end_i = ps_start_i + initial_window_size
        else:
            ps_end_i += 1

    return pd.DataFrame(power_sgmnts, index=idx)


def mean_chunk_power_sgmnts(series, max_deviation=10, window_size=10):
    watts = series.values
    ps_start_i = 0
    idx = [] # index for dataframe
    power_sgmnts = []
    n_chunks = int(watts.size / window_size)
    for chunk_i in range(1,n_chunks-2):
        ps_end_i = window_size * chunk_i
        ps = watts[ps_start_i:ps_end_i]
        next_chunk = watts[ps_end_i:ps_end_i+window_size]
        if (next_chunk.mean() > ps.mean() + max_deviation or
            next_chunk.mean() < ps.mean() - max_deviation):
            # We've come to the end of a candidate power segment
            idx.append(series.index[ps_start_i])
            power_sgmnts.append({'end': series.index[ps_end_i-1], 
                                 'power_stats': Normal(ps)})
            ps_start_i = ps_end_i

    return pd.DataFrame(power_sgmnts, index=idx)


def minimise_mean_deviation_power_sgmnts(series,
                                         max_deviation=20, # watts
                                         initial_window_size=30, # int: number of samples
                                         look_ahead=150, # int: number of samples
                                         max_ptp=1000 # max peak-to-peak watts
                                         ):
    watts = series.values
    ps_start_i = 0
    ps_end_i = initial_window_size
    idx = [] # index for dataframe
    power_sgmnts = []
    half_window = int(initial_window_size / 2)
    n = watts.size - look_ahead
    while True:
        if ps_end_i >= n:
            break

        # this is an initial chunk so test it's a sane
        # chunk by comparing the means of the left and right side
        # of this chunk
        ps = watts[ps_start_i:ps_end_i]
        if ps_end_i == ps_start_i + initial_window_size:
            halfway = ps_start_i + half_window
            left = watts[ps_start_i:halfway]
            right = watts[halfway:ps_end_i]
            if (math.fabs(left.min() - right.min()) > max_deviation or
                math.fabs(left.max() - right.max()) > max_deviation or
                ps.ptp() > max_ptp):
                ps_start_i += 1
                ps_end_i += 1
                continue

        # Now creep forwards from ps_end_i and find the index which
        # gives the lowest deviation of the mean of the power segment
        # against the mean of the look ahead chunk.
        min_deviation = np.finfo(np.float64).max
        i_of_lowest = None
        ps_mean = ps.mean()
        for look_ahead_i in range(ps_end_i+1, ps_end_i+look_ahead):
            ahead = watts[ps_end_i:look_ahead_i]
            # TODO: don't recalculate ahead.mean() from scratch every iteration
            deviation = math.fabs(ahead.mean() - ps_mean)
            if deviation <= min_deviation:
                min_deviation = deviation
                i_of_lowest = look_ahead_i

        # Now we've found the index of the minimum deviation
        if min_deviation <= max_deviation:
            ps_end_i = i_of_lowest
        else:
            # End of power segment
            idx.append(series.index[ps_start_i])
            power_sgmnts.append({'end': series.index[ps_end_i-1],
                                 'power_stats': Normal(ps)})
            ps_start_i = ps_end_i
            ps_end_i = ps_start_i + initial_window_size

    return pd.DataFrame(power_sgmnts, index=idx)


#############################################################################
# MERGE FEATURES
#############################################################################

def merge_features(pwr_sgmnts, decays, spike_histogram):
    """Associate features with each other to produce a list of 
    "signature power states".

    Returns:
        List of PowerStates.  Each records:
        * start: datetime of start of each power state
        * end: datetime of end of each power state
        * power_stats (Normal)
        * decay (float)
        * spike_histogram (2D np.ndarray): one col per bin

    """
    merged = []
    for start, pwr_seg in pwr_sgmnts.iterrows():
        pwr_state = PowerState(start=start, 
                               end=pwr_seg['end'],
                               power_stats=pwr_seg['power_stats'])

        # DECAYS:
        # Assume decays to be within some constant number of
        # seconds around the start.  Say 10 seconds. Set start time of
        # powerstate to be start time of decay.        
        max_time_diff = datetime.timedelta(seconds=10)
        i_of_nearest_decay = utils.find_nearest(decays, target=start, 
                                                max_time_diff=max_time_diff)
        if i_of_nearest_decay is not None:
            pwr_state.decay = decays.iloc[i_of_nearest_decay]['slope']
            pwr_state.start = decays.index[i_of_nearest_decay]
        
        # SPIKE HISTOGRAM:
        # Just take all.

        cropped_spike_hist = spike_histogram[(spike_histogram.index >=
                                              pwr_state.start) &
                                             (spike_histogram.index < 
                                              pwr_state.end)]
        pwr_state.spike_histogram = cropped_spike_hist.values

        merged.append(pwr_state)

    return merged
        
