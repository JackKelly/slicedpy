import feature_detectors as fd
import networkx as nx
import matplotlib.pyplot as plt
import Image

class Appliance(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.power_state_graph = nx.DiGraph()
        self.power_state_graph.add_node('off')

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
        return sig_power_states
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
        """Take the list of sig_power_states and merge these 
        into self.power_state_graph

        Args:
          ``sig_power_states`` (list of PowerStates)
        """
        
        prev_ps = 'off'
        for sps in sig_power_states:
            found_match = False
            for ps in self.power_state_graph.nodes():
                if ps != 'off' and ps.similar(sps):
                    ps.merge(sps.prepare_for_power_state_graph())
                    found_match = True
                    break
            if not found_match:
                ps = sps.prepare_for_power_state_graph()
                self.power_state_graph.add_node(ps)

            # Add edge
            self.power_state_graph.add_edge(prev_ps, ps)
            prev_ps = ps

        self.power_state_graph.add_edge(prev_ps, 'off')

        # Update count_per_run GMM for each power state:
        for ps in self.power_state_graph.nodes():
            if ps != 'off':
                ps.save_count_per_run()

    def draw_power_state_graph(self, png_filename='power_state_graph.png'):
        p = nx.to_pydot(self.power_state_graph)
        p.write_png(png_filename)
        im = Image.open(png_filename)
        im.show()

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
