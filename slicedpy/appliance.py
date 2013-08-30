import feature_detectors as fd
from powerstate import PowerState
import networkx as nx

class Appliance(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.power_state_graph = nx.DiGraph()
#        self.power_state_graph.add_node(PowerState('off'))

    def train_on_single_example(self, sig):
        """
        Call this multiple times if multiple examples are available.

        Args:
            sig (pda.Channel): example power signature of appliance
        """

        # Extract features.  Each returns a DataFrame.
        pwr_sgmnts = fd.min_max_power_sgmnts(sig.series)
        decays = fd.spike_then_decay(sig.series)
        spike_histogram, bin_edges = fd.spike_histogram(sig.series)

        # Now fuse features: i.e. associate features with each other
        # to come up with "signature power states".
        sig_power_states = fd.merge_features(pwr_sgmnts, decays, spike_histogram)
        return sig_power_states

        # Now take the sequence of sig power states and merge these into
        # the set of unique power states we keep for each appliance.
        ##### self.update_power_state_graph(sig_power_states)
        # Just figure out which power segments are
        # similar based just on power.  Don't bother trying to split
        # each power state based on spike histogram yet.
        # self.power_state_graph is NetworkX DiGraph.
        # self.power_state_graph node 0 is 'off'
        # If power never drops to 0W then self.power_state_graph and a node PowerState('standby')
        # and we should never be able to enter 'off' state??
        # each PowerState has these member variables:
        #   * duration: DataStore (GMM)
        #   * power: DataStore (Normal)
        #   * decay: DataStore (GMM)
        #   * spike_histogram: 2D DataStore (GMM), one col per bin 
        #     (don't bother recording bin edges, assume these remain constant
        #      in fact, put bin edges in a config.py file)
        #   * count_per_run = DataStore (GMM): number of times this power state is seen per run 

        # Graph edges:
        #   * diff between power segment mean: DataStore (GMM), 
        #   * forward diff: DataStore (GMM) <- actually, maybe don't bother with this
        #   * time between states^: DataStore (GMM)
        #   * mean power used between states^': float
        #   * 'probability': Float (or can we use DiGraph's 'weight' edge attribute?): per
        #     node, the probability of all *outbound* edges sum to 1.
        #
        # ^ these are used for handling the case where, for example, in the washing machine,
        # there are some really wild sections which are so wild that they are ignored
        # by the power segment detector.  But these sections still use power!
        #
        # update: 
        # * self.total_duration: DataStore (GMM)
        # * self.total_energy: DataStore (GMM)

        # Not sure if we need to update these...
        # * self.reliable_powerstate_transitions = [(0,1), (1,5)] - transitions which are always
        # * self.most_prominent_features: ordered list of 5 most
        #   reliable & prominent features (perhaps do this based on
        #   magnitude of 'diff between power states' for now?  Later we
        #   could try to find prominent features based on other
        #   features.  Alternatively, perhaps an appropriate classifier
        #   would select the most prominent features for free (e.g. a
        #   decision tree)).  i.e. features which are always present;
        #   then rank by saliency.


    def disaggregate(self, aggregate, pwr_sgmnts, decays, spike_histogram):
        """
        Incrementally build up candidate agg_power_states, as the union of all
        power states suggested by each feature extractor:
          1. find all transitions between pwr_sgmnts consistent with transitions
             seen for this appliance.  Require both +ve and -ve transitions
             (or maybe just require either / or.  Remember that we want to be 
             permissive to give the discrete optimisation something to do!)
             Create a list of candidate agg_power_states, with scores, durations and
             magnitude of power change (to tweak the energy estimation).
          2. find all ramps consistent with ramps for this appliance.  
             Add / modify agg_power_states.
          3. Same with spike histogram.

        Find all clusters of power states which appear within a self.total_duration.max
        period of time.  Then check that these clusters have legal state sequences.
        If they do then 

        1. Find any of the most prominent features.  Then check if
           other prominent features are present within self.max_duration.

        2. If they are then we have identified appliance.  Then track each power
           state (to get an accurate measure of energy used per run).
        """

    def refine(self, aggregate):
        """
        Refine general models to each home 

        First pass: find appliances and appliance power states based on
        general model (this will necessarily be permissive).  Ignore power
        states where other appliances are producing signals in any feature
        detector (this might be too strict?  Maybe just don't re-train feature
        detector if that feature detector is changing due to another
        appliance?)  Redefine power states based on "quiet" power states found
        in aggregate data.  Cite Oliver Parsons.
        """
