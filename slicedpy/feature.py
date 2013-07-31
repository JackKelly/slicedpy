from bunch import Bunch

class Feature(Bunch):
    """All feature detectors output a list of Feature objects.
    The idea is that all Features must have a start and an end timestamp
    plus zero or more other parameters specific to that feature detector.

    Attributes:
        start: pandas.datetime64['s']
        end: pandas.datetime64['s']
    """
    def __init__(self, start, end, **kwds):
        self.start = start
        self.end = end
        super(Feature, self).__init__(**kwds)
