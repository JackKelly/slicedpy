from __future__ import print_function, division
from slicedpy.bunch import Bunch
from slicedpy.datastore import DataStore
from slicedpy.normal import Normal
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta
import numpy as np

class Edge(Bunch):
    """Represents edges on power state graph.

    Attributes:
      * power_segment_diff (DataStore(n_columns=2, model=NearestNeighbors()):
        Stores difference (power and time) between the two power segments.
      * edge_power (DataStore(model=Normal()): average power consumed
        _between_ power segments. (used to estimate energy used between power
        segments)
      * edge_fwd_diff (DataStore(model=NearestNeighbors()): the largest "spike"
        in the raw power data observed around the start of the receiving node)
    """

    def __init__(self, **kwds):
        super(Bunch, self).__init__(**kwds)
        self.power_segment_diff = DataStore(n_columns=2, model=NearestNeighbors())
        self.edge_power = DataStore(model=Normal())
        self.edge_fwd_diff = DataStore(model=NearestNeighbors())

    def update(self, sps, prev_sps, sig):
        """
        Args:
          * sps (PowerState): signature power segment of node receiving this edge
          * prev_sps (PowerState): the signature power segment assigned to the
            node from which this edge begins.
          * sig (pda.Channel): raw power data
        """

        edge_dur = (sps.start - prev_sps.end).total_seconds()
        edge_pwr = sig.crop(prev_sps.end, sps.start).joules() / edge_dur
        watts_near_start = sig.crop(sps.start-timedelta(seconds=6),
                                    sps.start+timedelta(seconds=5))
        fdiff = watts_near_start.series.diff().dropna()
        i_of_largest_fdiff = fdiff.abs().argmax()
        edge_fwd_diff = fdiff.iloc[i_of_largest_fdiff]
        sps_diff = sps.power.get_model().mean - prev_sps.power.get_model().mean
        
        self.power_segment_diff.append(np.array([sps_diff, edge_dur], ndmin=2))
        self.edge_power.append(edge_pwr)
        self.edge_fwd_diff.append(edge_fwd_diff)

        print("edge from", prev_sps.power.get_model().mean,
              "to", sps.power.get_model().mean,
              "=", self)

    def __str__(self):
        s = ""
        psd = self.power_segment_diff.data[-1]
        s += ("last power_segment_diff: power={:.2f}W, time={:.2f}s\n"
              .format(psd[0], psd[1]))
        return s