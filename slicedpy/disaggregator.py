from sklearn import tree
import subprocess

class Disaggregator(object):
    """
    Responsibilities of this class:
    -------------------------------

    Training:
    * responsible for storing list of all known appliances
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
        self.reset()

    def reset(self):
        self.appliances = []
        self.power_state_decision_tree = tree.DecisionTreeClassifier()

    ##########################################################################
    # DECISION TREES
    ##########################################################################

    def train_decision_tree(self, appliances):
        """
        Args:
          * appliances (list of Appliance objects)
        """
        self.appliances = appliances

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
        
