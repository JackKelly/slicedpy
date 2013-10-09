from __future__ import print_function, division
import subprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import mixture
import slicedpy.feature_detectors as fd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from slicedpy.plot import plot_data_and_model

class Disaggregator(object):
    """
    Responsibilities of this class:
    -------------------------------

    Training:
    * responsible for storing list of all known appliances
    """

    def __init__(self):
        self.appliances = []

BIN_WIDTH = 5 # watts. Must be an int.
MIN_FWD_DIFF = 1 # watts


def get_bins(data):
    """
    Args:
      * data (np.ndarray)
    """
    start = math.floor(data.min())
    start = start if start % BIN_WIDTH == 0 else start - (start % BIN_WIDTH)
    start -= MIN_FWD_DIFF
    start = int(start)

    stop = math.ceil(data.max())
    stop = stop if stop % BIN_WIDTH == 0 else stop + (BIN_WIDTH - (stop % BIN_WIDTH))
    stop += MIN_FWD_DIFF + BIN_WIDTH
    stop = int(stop)

    neg_bins = np.arange(start=start, stop=-MIN_FWD_DIFF, step=BIN_WIDTH)
    pos_bins = np.arange(start=MIN_FWD_DIFF, stop=stop, step=BIN_WIDTH)
    bin_edges = np.concatenate([neg_bins, [-MIN_FWD_DIFF], pos_bins])
    n_negative_bins = len(neg_bins)
    return bin_edges, n_negative_bins


class BayesDisaggregator(Disaggregator):

    def __init__(self):
        super(BayesDisaggregator, self).__init__()

    def _fit_p_fwd_diff(self, aggregate, plot=True):
        """Estimate the probability density function for P(forward diff).

        Args:
          * aggregate (pda.Channel or np.ndarray)
        """

        # TODO:
        # * merge spikes

        if isinstance(aggregate, np.ndarray):
            fwd_diff = np.diff(aggregate)
        else:
            fwd_diff = aggregate.diff_ignoring_long_outages()

        fwd_diff = fwd_diff[np.fabs(fwd_diff) >= MIN_FWD_DIFF]

        self._bin_edges, self._n_negative_bins = get_bins(fwd_diff)
        density, bin_edges = np.histogram(fwd_diff, bins=self._bin_edges, 
                                          density=True)

        # Treat the bins as discrete values and come up with a
        # 'probability mass' for each bin:
        self._prob_mass = density * BIN_WIDTH

        # Plot
        if plot:
            ax = plt.gca()
            ax.bar(bin_edges[:-1], self._prob_mass)
            plt.show()
            return ax

    def _p_fwd_diff(self, fwd_diff, print_debug_info=False):
        """The probability of the forward diff, as calculated previously
        using _fit_p_fwd_diff().  If fwd_diff lies in the middle of a bin
        then return the probability mass associated with that bin otherwise
        return the weighted average of that bin and the neighbouring bin.

        Args:
          * fwd_diff (float): watts

        Returns:
          * p_fwd_diff (float): [0,1]
        """
        # take weighted average from nearest 2 bins.
        # careful near zero and at extremes
        if (abs(fwd_diff) < MIN_FWD_DIFF or
            fwd_diff >= self._bin_edges[-1] or
            fwd_diff < self._bin_edges[0]):
            return 0.0

        # Calculate the bin index corresponding to fwd_diff
        if fwd_diff < 0:
            bin1_i = (math.floor((fwd_diff + MIN_FWD_DIFF) / BIN_WIDTH) +
                     self._n_negative_bins)
        else:
            bin1_i = (math.floor((fwd_diff - MIN_FWD_DIFF) / BIN_WIDTH) +
                     self._n_negative_bins + 1)
        bin1_i = int(bin1_i)

        if abs(fwd_diff) <= MIN_FWD_DIFF + (BIN_WIDTH / 2):
            return self._prob_mass[bin1_i]

        n_bins = len(self._bin_edges) - 1
        position_in_bin1 = (fwd_diff - self._bin_edges[bin1_i]) / BIN_WIDTH
        if position_in_bin1 == 0.5: # exactly half-way in bin1_i
            return self._prob_mass[bin1_i]
        elif position_in_bin1 > 0.5:
            if bin1_i == n_bins - 1: # in top bin
                return self._prob_mass[bin1_i]
            bin2_i = bin1_i + 1
        else: # position_in_bin1 < 0.5
            if bin1_i == 0:
                return self._prob_mass[bin1_i]
            bin2_i = bin1_i - 1

        # Take weighted average of bin1_i and bin2_i
        proportion_of_bin1 = 1 - abs(position_in_bin1 - 0.5)
        proportion_of_bin2 = 1 - proportion_of_bin1

        p_fwd_diff = ((proportion_of_bin1 * self._prob_mass[bin1_i]) +
                       proportion_of_bin2 * self._prob_mass[bin2_i])

        if print_debug_info:
            fmt = '{:.1f}\n'
            print(('fwd_diff='+fmt+'position_in_bin1='+fmt+'prop_of_bin1='+fmt+
                   'prop_of_bin2='+fmt+'p_fwd_diff='+fmt)
                  .format(fwd_diff, position_in_bin1, proportion_of_bin1,
                          proportion_of_bin2, p_fwd_diff))

        return p_fwd_diff

    def train(self, aggregate, appliances):
        """
        Args:
          * aggregate (pda.Channel)
          * appliances (list of slicedpy.Appliance objects)
        """
        pass


class KNNDisaggregator(Disaggregator):

    ##########################################################################
    # K-NEAREST NEIGHBOURS
    ##########################################################################

    def __init__(self):
        super(KNNDisaggregator, self).__init__()

    def train_knn(self, appliances):
        """
        Args:
          * appliances (list of Appliance objects)
        """
        self.appliances = appliances
        self.power_seg_diff_knn = KNeighborsClassifier(n_neighbors=1,
                                                       weights='distance')
        X = [] # feature matrix
        Y = [] # labels
        for app in self.appliances:
            X_app, Y_app = app.get_inbound_edge_feature_matrix()
            X.extend(X_app)
            for edge in Y_app:
                Y.append({'appliance':app, 'edge':edge})

        self.power_seg_diff_knn.fit(X, Y)

    def disaggregate(self, aggregate, return_power_segments=False):
        """
        Args:
          * aggregate (pda.Channel)
          * return_power_states (bool): Default=False. Set to true to
            return pd.DataFrame, power_segments

        Returns:
          * pd.DataFrame. Each row is an appliance hypothesis. Columns:
            - index (datetime): start of appliance hypothesis
            - end (datetime): end of appliance hypothesis
            - appliance (slicedpy.Appliance)
            - likelihood (float)
        """
        pwr_sgmnts = fd.min_max_power_sgmnts(aggregate.series)
        output = []
        rng = []
        MAX_DIST = 100 # watts

        prev_pwr_seg = pwr_sgmnts.iloc[0]
        for start, pwr_seg in pwr_sgmnts.iloc[1:].iterrows():
            pwr_seg_mean_diff = (pwr_seg['power'].get_model().mean -
                                 prev_pwr_seg['power'].get_model().mean)
#            pwr_seg_time_diff = (start - prev_pwr_seg['end']).total_seconds()

            test_data = np.array([pwr_seg_mean_diff, 1])
            dist, ind = self.power_seg_diff_knn.kneighbors(test_data)
            if dist.min() < MAX_DIST:
                predicted = self.power_seg_diff_knn.predict(test_data)[0]
                
                output.append({'end': pwr_seg['end'],
                               'appliance': predicted['appliance']})
                rng.append(start)

            prev_pwr_seg = pwr_seg

        df = pd.DataFrame(output, index=rng)

        if return_power_segments:
            return df, pwr_sgmnts
        else:
            return df

class DTDisaggregator(Disaggregator):

    ##########################################################################
    # DECISION TREES
    ##########################################################################

    """
    * creates decision tree for all power states

    Disaggregation:
    EITHER READ TREE FORWARDS OR BACKWARDS:

    FORWARD:
    * split aggregate into power segments.  Then classify these
      using decision tree (ignoring duration for now).  Then use HMM trained
      for each appliance.

    BACKWARDS:
    For each appliance:
      For each power state within appliance:
        * reads the decision tree backwards to find most efficient way to find 
          each power state.
        * then tries to find complete power segments
    """


    def __init__(self):
        super(DTDisaggregator, self).__init__()

    def train_decision_tree(self, appliances):
        """
        Args:
          * appliances (list of Appliance objects)
        """
        self.appliances = appliances
        self.power_state_decision_tree = tree.DecisionTreeClassifier()

        # training data:
        X = [] # list of feature vectors
        Y = [] # labels
        
        for i, appliance in enumerate(self.appliances):
            X.extend(appliance.feature_matrix)
            n = len(appliance.feature_matrix)
            Y.extend(zip([appliance.label]*n, appliance.feature_matrix_labels))
#            Y.extend([appliance.label]*n)

        self.power_state_decision_tree.fit(X, Y)
        
    def draw_tree(self, base_filename='decision_tree'):
        dot_filename = base_filename + '.dot'
        with open(dot_filename, 'w') as f:
            f = tree.export_graphviz(self.power_state_decision_tree, out_file=f,
                                     feature_names=['W', 'dur', 'slope', 
                                                    'sh0', 'sh1', 'sh2', 'sh3', 
                                                    'sh4', 'sh5', 'sh6', 'sh7'])
        subprocess.call(['dot', '-Tpdf', dot_filename, '-o', base_filename+'.pdf'])
        subprocess.call(['evince', base_filename+'.pdf'])
        
