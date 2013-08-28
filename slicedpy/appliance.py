import feature_detectors as fd
from powerstate import PowerState

class Appliance(object):

    def __init__(self):
        self.unique_power_states = [PowerState('off')]

    def train_on_single_example(self, sig):
        """
        Call this multiple times if multiple examples are available.

        Args:
            sig (pda.Channel): example power signature of appliance
        """

        # Extract features.  Each returns a DataFrame.
        pwr_sgmnts = fd.min_max_power_sgmnts(sig.series)
        decays = fd.spike_then_decay(sig.series)
        spike_histogram = fd.spike_histogram(sig.series)

        # Now fuse features: i.e. associate features with each other
        # to come up with "signature power states".
        sig_power_states = fd.sig_power_states(pwr_sgmnts, decays, spike_histogram)
        # DECAYS: 
        # Assume decays to be within some constant number of
        # seconds around the start.  Say 10 seconds. Set start time of
        # powerstate to be start time of decay.
        # SPIKE HISTOGRAM:
        # Just take all.
        # RETURNS
        # DataFrame.  One row per sig power state.  Cols:
        # * index = datetime of start of each power state
        # * end (np.datetime64? or datetime.datetime?) = datetime of end of each power state
        # * decay (float)
        # * spike_histogram (DataFrame: just a copy of spike_histogram masked
        #   by the start and end times of the powerstate.)

        # Now take the sequence of sig power states and merge these into
        # the set of unique power states we keep for each appliance.
        self.update_unique_power_states(sig_power_states)
        # Just figure out which power segments are
        # similar based just on power.  Don't bother trying to split
        # each power state based on spike histogram yet.
        # self.unique_power_states is a list of PowerStates where:
        # self.unique_power_states[0] is 'off'
        # If power never ever drops to 0W then self.unique_power_states[1] == PowerState('standby')
        # and we should never be able to enter 'off' state.
        # each PowerState has these member variables:
        #   * duration: GMM
        #   * power: GMM, 
        #   * decay: GMM, 
        #   * spike_histogram: list of GMMs, one per bin (don't bother recording bin edges,
        #     assume these remain constant)
        #   * count_per_run = GMM: number of times this power state is seen per run 
        #   * next_states = {1: {'diff between power states': GMM, 
        #                        'forward diff': GMM, 
        #                        'probability': Float}}
        #
        # Also store all raw training data in each PowerState.  This is necessary
        # so that we can re-fit GMMs when new signature examples are provided, or 
        # when refining appliance models for a house.  It will also allow us to 
        # experiment with classifiers which can handle all data.  And also to 
        # draw plots to manually check the fit of GMMs to the data.
        #   * _raw_durations = list of timedeltas
        #   * _raw_powers = list of floats
        #   * _raw_decays = list of floats
        #   * _raw_spike_histograms = list of np.ndarray
        #   * _raw_counts_per_run = list of ints

        # update: 
        # * self._raw_total_durations = list of timedeltas
        # * self._raw_total_energies = list of floats (kWh)
        # * self.total_duration (GMM)
        # * self.total_energy (GMM)

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
        detector (this might be too strict?  Maybe just don’t re-train feature
        detector if that feature detector is changing due to another
        appliance?)  Redefine power states based on “quiet” power states found
        in aggregate data.  Cite Oliver Parsons.
        """
