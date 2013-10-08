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

BIN_WIDTH = 5 # watts
MIN_FWD_DIFF = 1 # watts


def get_bins(data):
    start = math.floor(data.min())
    start = start if start % BIN_WIDTH == 0 else start - (start % BIN_WIDTH)
    start -= MIN_FWD_DIFF

    stop = math.ceil(data.max())
    stop = stop if stop % BIN_WIDTH == 0 else stop + (BIN_WIDTH - (stop % BIN_WIDTH))
    stop += MIN_FWD_DIFF + BIN_WIDTH

    neg_bins = np.arange(start=start, stop=-MIN_FWD_DIFF, step=BIN_WIDTH)
    pos_bins = np.arange(start=MIN_FWD_DIFF, stop=stop, step=BIN_WIDTH)
    bins = np.concatenate([neg_bins, [-MIN_FWD_DIFF], pos_bins])
    return bins


class BayesDisaggregator(Disaggregator):

    def __init__(self):
        super(BayesDisaggregator, self).__init__()

    def _fit_p_foward_diff(self, aggregate, plot=True):
        """Estimate the probability density function for P(forward diff).

        Args:
          * aggregate (pda.Channel)
        """

        # TODO: 
        # * filter out diffs where gap is > sample period.
        # * merge spikes

        fwd_diff = aggregate.series.diff().dropna().values
        fwd_diff = fwd_diff[np.fabs(fwd_diff) >= MIN_FWD_DIFF]

        self._bins = get_bins(fwd_diff)
        self._density, bin_edges = np.histogram(fwd_diff, bins=self._bins, 
                                                density=True)
        # Plot
        ax = plt.gca()
        ax.plot(bin_edges[:-1], self._density)
        plt.show()
        return ax

    def _p_fwd_diff(self, fwd_diff):
        # take weighted average from nearest 2 bins.
        # divide by BIN_WIDTH
        # careful near zero and at extremes
        pass

    def _old_fit_p_foward_diff(self, aggregate, plot=True):
        """Estimate the probability density function for P(forward diff).

        Args:
          * aggregate (pda.Channel)
        """
#        aggregate = aggregate.crop('2013/6/1','2013/6/7')
        # TODO: filter out diffs where gap is > sample period.
        fwd_diff = aggregate.series.diff().dropna().values
        fwd_diff = fwd_diff[np.fabs(fwd_diff) > 20]

        # find best number of components for GMM:
        lowest_bic = np.inf
        best_n_components = None
        best_cv_type = None
#        cv_types = ['spherical', 'tied', 'diag', 'full']
        cv_types = ['full']
        for cv_type in cv_types:
            for n_components in range(19,30):
                print('Trying n_components={:d}, cv_type={:s}.'
                      .format(n_components, cv_type))
                gmm = mixture.GMM(n_components=n_components, 
                                  covariance_type=cv_type)
                gmm.fit(fwd_diff)
                bic = gmm.bic(fwd_diff)
                if bic < lowest_bic:
                    print('  this model had the lowest BIC ({}) so far.'
                          .format(bic))
                    lowest_bic = bic
                    best_n_components = n_components
                    best_cv_type = cv_type

        self._p_fwd_diff = mixture.GMM(n_components=best_n_components, 
                                       covariance_type=cv_type)
        self._p_fwd_diff.fit(fwd_diff)

#        self._p_fwd_diff = mixture.DPGMM(n_components=20, covariance_type='full')
#        self._p_fwd_diff.fit(fwd_diff)

        if plot:
            print('Plotting...')
            return plot_data_and_model(fwd_diff, self._p_fwd_diff)


    def train(self, aggregate, appliances):
        """
        Args:
          * aggregate (pda.Channel)
          * appliances (list of slicedpy.Appliance objects)
        """
        


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
        
