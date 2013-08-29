from __future__ import print_function, division
from bunch import Bunch
import copy
from slicedpy.normal import Normal


class PowerState(Bunch):
    """
    A washing machine might have three power states: washing, heating,
    spinning.  PowerStates do not have a start
    or an end.

    Attributes:
      * features: a dict of Normals, usually including: 
        {duration: Normal(), # seconds
         power: Normal(), # watts
         ramp: None or Normal(),
         spike_histogram: [Normal, Normal, Normal, Normal, Normal]
        }
    """
    def __init__(self, **kwds):
        self.features = {}
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
            new_ps = PowerState(sig_power_segment=sps)
            unique_pwr_states.append(new_ps)
            mapped_sig_pwr_sgmnts[sps_i].power_state = len(unique_pwr_states)-1

    return unique_pwr_states, mapped_sig_pwr_sgmnts
