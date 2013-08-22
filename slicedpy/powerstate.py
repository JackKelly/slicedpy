from __future__ import print_function, division
from bunch import Bunch

class PowerState(Bunch):
    def __init__(self, signature_power_state=None, **kwds):
        if signature_power_state is not None:
            self.mean = signature_power_state.mean
            self.size = signature_power_state.size
            self.var = signature_power_state.var
        super(PowerState, self).__init__(**kwds)
