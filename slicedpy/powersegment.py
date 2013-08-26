from __future__ import print_function, division
from feature import Feature
from powerstate import PowerState

class PowerSegment(Feature, PowerState):
    """A ``power segment`` is a segment of an appliance signatures which
    roughly corresponds to a single ``mode`` of that appliance.  Power segments
    each have start and end indices.  For example, a washing machine
    might have many (10?) power segments.

    PowerSegments are identified in a signature using the 
    ``feature_detectors.*_power_sgmnt()`` functions.

    PowerSegments are generalised to :class:`PowerStates`.
    """

    def __init__(self, watts=None, **kwds):
        """
        Args
          * watts (np.ndarray): Power data for this power segment. 
            ``watts.size`` must equal ``end - start``.
        """
        if watts is not None and watts.size != self.size:
            raise ValueError('watts must only contain data for'
                             ' this power segment!')

        super(PowerSegment, self).__init__(values=watts, **kwds)
        PowerState.__init__(self)


    def add_spike_then_decay(self, stds):
        pass
