from __future__ import print_function, division
import numpy as np
from plot import plot_data_and_model

class DataStore(object):
    """Stores 1D array and allows for model estimation and plotting.

    Attributes:
      * data (np.ndarray)
      * history (list): indices into `data` recording when `append` is called.
      * model: object for estimating statistical model of data
    """

    def __init__(self, n_columns=1, model=None):
        """
        Args:
          * model: an object with API similar to scikit-learn's 
            (i.e. with methods `.fit()` etc). e.g. and instance of class
            slicedpy.normal.Normal or sklearn.mixture.GMM
          * n_columns (int): Optional. Default=1.
        """
        self.n_columns = n_columns
        self.model = model
        self.clear()

    def clear(self):
        shape = (0, self.n_columns) if self.n_columns > 1 else (0,)
        self.data = np.empty(shape)
        self.history = []
        self._model_is_stale = False

    def append(self, new_data):
        """Appends `new_data` to `self.data` and fits model.

        Args:
          * new_data (np.ndarray or float or int)
        """
        if new_data is None:
            return
        elif isinstance(new_data, (float, int, np.floating, np.integer)):
            new_data = np.array([new_data])
        elif isinstance(new_data, np.ndarray):
            if new_data.shape[0] == 0:
                return # can't do anything useful with no data
        else:
            raise TypeError('new_data must be a scalar or a numpy.ndarray')

        self.history.append(self.data.size)
        self.data = np.append(self.data, new_data, axis=0)
        self._model_is_stale = True

    def get_model(self):
        if self._model_is_stale:
            self.fit()
        return self.model

    def fit(self):
        if self.model is None:
            raise Exception('self.model is None! '
                            ' It must be set before calling fit()!')

        if self.data.shape[0] == 0:
            raise Exception('cannot fit model when we have no data!')
        elif self.data.shape[0] == 1:
            # We cannot fit model to a single value
            # so if we have a single value then repeat it twice.
            data = np.append(self.data, self.data, axis=0) 
        else:
            data = self.data

        self.model.fit(data)
        self._model_is_stale = False

    def extend(self, other):
        if other is None:
            return
        if self.n_columns != other.n_columns:
            raise Exception('self.n_columns != other.n_columns')
        self.history.extend(other.history)
        self.append(other.data)

    def plot(self, ax, **kwds):
        """Plots comparison of data histogram and model.

        Args:
          * ax (matplotlib.Axes)
        """
        ax = plot_data_and_model(ax=ax, data=self.data, model=self.model, **kwds)
        return ax


def test_fitting_and_plotting():
    from sklearn import mixture
    import matplotlib.pyplot as plt

    model = mixture.GMM(n_components=1)
    ds = DataStore(model=model)
    ds.append(np.random.randn(100,1))
    ds.append(10+np.random.randn(200,1))

    fig, ax = plt.subplots()
    ds.plot(ax)
    plt.show()
    print("Done")
