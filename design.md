## Training appliance models

### Appliance.train_on_single_example(signature) 

1. Looks for power segments within signature (done).  Returns
   FeatureList of Features each with features['power'] set.

2. Find decays (returns FeatureList of Features each with
   features['decay'] set) 

3. Find spike histogram (returns FeatureList of Features each with
   feature = {'spike_histogram': {(0,7): Normal, (8,40): Normal...}} )

4. pwr_segs = pwr_segs.assimilate_feature(decay, how='one near start')
   Just assume it'll be within some constant number of samples around the
   start.  Say 10 samples.  Don't modify the start time of the Feature.
  
5. pwr_segs = pwr_segs.assimilate_feature(spike_histogram, how='average all') 
   Take all the Features (in this case, 1-minute windows) and average them all.

6. self.assimilate_power_segments(ps).  Update self.power_states.
   Just figure out which power segments are similar based just on
   power.  Don't bother trying to split each power state based on
   spike histogram yet. Returns “map of power_segments to power
   states” and updates self.power_states (a list of unique PowerStates
   (where entry zero is 'off') each with: features = {duration:
   Normal(), power: Normal(), slope: Normal(), spike_histogram:
   {(0,7): Normal(), (8,40): Normal()...}  count_per_run = Normal() #
   number of times this power state is seen per run next_states = {1:
   {'diff between power states': Normal, 'forward diff': Normal,
   'probability': Float}} 1. Later, we could then use power_state_map
   to train a classifier like an SVM to recognise each power state.

7. update: 
   * self.total_duration (Normal) 
   * self.total_energy (Normal) 

10. self.most_prominent_features: ordered list of 5 most
   reliable & prominent features (perhaps do this just for magnitude
   of 'diff between power states' for now?  Later we could try to find
   prominent features based on other features.  Alternatively, perhaps
   an appropriate classifier would select the most prominent features
   for free (e.g. a decision tree)).  i.e. features which are always
   present; then rank by saliency.

Play around with the output from this Appliance creation system.  Draw
graphical models showing transition between states with probabilities,
diff between power states and forward diff annotated on edges.


## Disaggregation 

1. Find any of the most prominent features.  Then check if
   other prominent features are present within appliance.max_duration.

2. If they are then we have identified appliance.  then track each power
   state (to get an accurate measure of energy used per run)


## Refine general models to each home 

First pass: find appliances and appliance power states based on
general model (this will necessarily be permissive).  Ignore power
states where other appliances are producing signals in any feature
detector (this might be too strict?  Maybe just don’t re-train feature
detector if that feature detector is changing due to another
appliance?)  Redefine power states based on “quiet” power states found
in aggregate data.
