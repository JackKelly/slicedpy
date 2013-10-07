import subprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import slicedpy.feature_detectors as fd
import pandas as pd
import numpy as np

class Disaggregator(object):
    """
    Responsibilities of this class:
    -------------------------------

    Training:
    * responsible for storing list of all known appliances
    """

    def __init__(self):
        self.appliances = []


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
        
