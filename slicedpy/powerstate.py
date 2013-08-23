from __future__ import print_function, division
from bunch import Bunch
import copy
import slicedpy.stats as spstats


class PowerState(Bunch):
    def __init__(self, signature_power_state=None, **kwds):
        if signature_power_state is not None:
            self.mean = signature_power_state.mean
            self.size = signature_power_state.size
            self.var = signature_power_state.var
        super(PowerState, self).__init__(**kwds)


def merge_pwr_sgmnts(signature_pwr_segments):
    """Merge signature power segments into a set of unique power states.

    Args:
      * signature_pwr_segments (list of :class:`Features`; each with a 
        ``start``, ``end``, ``mean``, ``var``, ``size``)

    Returns:
      ``unique_pwr_states``, ``signature_pwr_states``
      * ``unique_pwr_states`` is a list of unique :class:`PowerState`s
      * ``signature_pwr_states`` is a copy of ``signature_pwr_segments``
        where each item has an additional field ``power_state`` (int) 
        which is the index into ``unique_pwr_states`` for that power segment.
        That is, the ``power_state`` field maps from the power segment to
        a single power state.
    """

    unique_pwr_states = []
    signature_pwr_states = copy.copy(signature_pwr_segments)
    for sps_i, sps in enumerate(signature_pwr_segments):
        match_found = False
        for ups_i, ups in enumerate(unique_pwr_states):
            if spstats.similar_mean(sps, ups): 
                mean_ups = spstats.rough_mean_of_two_normals(sps, ups)
                unique_pwr_states[ups_i] = PowerState(mean_ups)
                match_found = True
                signature_pwr_states[sps_i].power_state = ups_i
                break
        if not match_found:
            new_ps = PowerState(signature_power_state=sps)
            unique_pwr_states.append(new_ps)
            signature_pwr_states[sps_i].power_state = len(unique_pwr_states) - 1

    return unique_pwr_states, signature_pwr_states
