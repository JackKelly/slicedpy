from __future__ import print_function, division
from bunch import Bunch
import copy
import slicedpy.stats as spstats


class PowerState(Bunch):
    """A ``PowerState`` is a generalisation of :class:`PowerSegment`.
    A washing machine might have three power states: washing, heating,
    spinning.  PowerStates, unlike PowerSegments, do not have a start
    or an end.
    """
    def __init__(self, signature_power_state=None, **kwds):
        if signature_power_state is not None:
            self.mean = signature_power_state.mean
            self.size = signature_power_state.size
            self.var = signature_power_state.var
        super(PowerState, self).__init__(**kwds)


def merge_pwr_sgmnts(signature_pwr_segments):
    """Merge signature :class:`PowerSegment`s into a list of 
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
            new_ps = PowerState(signature_power_state=sps)
            unique_pwr_states.append(new_ps)
            mapped_sig_pwr_sgmnts[sps_i].power_state = len(unique_pwr_states)-1

    return unique_pwr_states, mapped_sig_pwr_sgmnts
