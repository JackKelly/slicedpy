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

        # Now take the sequence of sig power states and merge these into
        # the set (list) of unique power states we keep for each appliance.
        self.update_power_state_graph(sig_power_states)
#        self.plot_power_state_graph()
        return
        # Just figure out which power segments are
        # similar based just on power.  Don't bother trying to split
        # each power state based on spike histogram yet.
        # self.power_state_graph is NetworkX DiGraph.
        # self.power_state_graph node 0 is 'off'
        # If power never drops to 0W then self.power_state_graph and a node PowerState('standby')
        # and we should never be able to enter 'off' state??
        # each PowerState has these member variables [see PowerState class]:

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

    def update_power_state_graph(self, sig_power_states):
        """
        Args:
          ``sig_power_states`` (list of PowerStates)
        """

    # Repurpose this old code...
    #def merge_pwr_sgmnts(signature_pwr_segments):
        """
        THIS FUNCTION IS CURRENTLY BROKEN PENDING REFACTORING!!!

        Merge signature :class:`PowerSegment`s into a list of 
        unique :class:`PowerState`s.

        Args:
          * signature_pwr_segments (list of :class:`PowerSegments`; each with a 
            ``start``, ``end``, ``mean``, ``var``, ``size``)

        Returns:
          ``unique_pwr_states``, ``mapped_sig_pwr_sgmnts``
          * ``unique_pwr_states`` is a list of unique :class:`PowerState`s
          * ``mapped_sig_pwr_sgmnts`` is a copy of ``signature_pwr_segments``
            where each item has an additional field ``power_state`` (int) 
            which is the index into ``unique_pwr_states`` for that power segment.
            That is, the ``power_state`` field maps from the power segment to
            a single power state.
        """

        unique_pwr_states = []
        mapped_sig_pwr_sgmnts = copy.copy(signature_pwr_segments)
        for sps_i, sps in enumerate(signature_pwr_segments):
            match_found = False
            for ups_i, ups in enumerate(unique_pwr_states):
                if spstats.similar_mean(sps, ups): 
                    mean_ups = spstats.rough_mean_of_two_normals(sps, ups)
                    unique_pwr_states[ups_i] = PowerState(mean_ups)
                    match_found = True
                    mapped_sig_pwr_sgmnts[sps_i].power_state = ups_i
                    break
            if not match_found:
                new_ps = PowerState(sig_power_segment=sps)
                unique_pwr_states.append(new_ps)
                mapped_sig_pwr_sgmnts[sps_i].power_state = len(unique_pwr_states)-1

        return unique_pwr_states, mapped_sig_pwr_sgmnts


    def disaggregate(self, aggregate, pwr_sgmnts, decays, spike_histogram):
        """Find all possible single power states
        -------------------------------------

        Each appliance has a set of power states.  For each power state we 
        have a set of "ways in" and a set of "ways out".  These "ways in" and
        "ways out" comprise the *differences between* power states (like Hart's
        edge detector).  Find all possible single power states (don't worry
        about sequence for now); *including* overlapping candidates (remember 
        that we want to be permissive to give the discrete optimisation
        something to do!):
        
        For each power state in self.power_state_graph:
          1) Find all possible "ways in": find all power state transitions 
             between the min and max of any "ways in" in the aggregate. (If we
             want to be really permissive then also find candidates based just
             on decays / spike histogram: e.g. if
             this state starts with a ramp then find all ramps consistent with
             this state and add / merge these with "ways in" found just from
             power state transitions... but let's not do that to start with; just
             find candidates based on power state transitions).
          2) For each "candidate way in", find all power state transitions
             between the min and max of any "ways out", within the min and
             max duration of the power state.  If there are no "ways out" for
             this "way in" then ignore this "way in".  If there are multiple
             "ways out" then create multiple (overlapping) candidate power
             states. For each candidate power state, output:
               * the confidence (based on way in, way out, duration, 
                 decay, spike hist)
               * the start & end times
               * an index into self.power_state_graph.nodes identifying this 
                 power state.
               * magnitude of power change (to tweak the energy estimation).


        Handle overlapping power states
        -------------------------------

        Several (mutually exclusive) options:
          1) Simplest (do this first): identify groups of overlapping
             candidate power states.  For each group, delete all but
             the most confident. (should we only consider each class
             of power state alone? If so then what about overlaps
             between different types of power state? Maybe simplest to
             consider overlaps between any type of power state and
             select only the most confident?)
          2) I think Hart mentioned something about modifying the Viterbi 
             algorithm to handle inserted symbols.  Check this out.
          3) Or for each segment with overlaps: enumerate all possible "detours"
             through this segment using non-overlapping candidates, and then find
             the most likely route *for each overlapping segment*, and select that one.


        Decoding
        --------
        For two state appliances we're done.

        For multi-state appliances: enumerate every permutation and
        calculate the joint prob of the Markov chain.

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
