from sklearn import tree

class Disaggregator(object):
    """
    Responsibilities of this class:
    -------------------------------

    Training:
    * responsible for storing list of all known appliances
    * creates decision tree for all power states

    Disaggregation:
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

    def train(self, appliances):
        """
        Args:
          * appliances (list of Appliance objects)
        """
        self.appliances = appliances

        # training data:
        X = [] # list of feature vectors
        Y = [] # labels
        
        for i, appliance in enumerate(self.appliances):
            for ps in appliance.power_state_graph.nodes():
                if ps == 'off':
                    continue
                X.append(ps.get_feature_vector())
                Y.append((i, ps))

        self.power_state_decision_tree.fit(X, Y)
                
