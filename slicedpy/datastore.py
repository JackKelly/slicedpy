from __future__ import print_function, division
import numpy as np

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

    def append(self, new_data):
        """Appends `new_data` to `self.data` and fits model.

        Args:
          * new_data (np.ndarray or float or int)
        """
        if isinstance(new_data, (float, int)):
            new_data = np.array([new_data])
        self.history.append(self.data.size)
        self.data = np.append(self.data, new_data, axis=0)
        if self.data.shape[0] == 0:
            return # cannot fit model when we have no data!

        def pad_single_value(data):
            # helper function. We cannot fit model to a single value
            # so if we have a single value then repeat it twice.
            return np.append(data, data, axis=0) if data.shape[0] == 1 else data

        # Fit model...
        if self.model is None:
            raise Exception('self.model is None! '
                            ' It must be set before calling fit()!')
        try:
            data = pad_single_value(new_data)
            self.model.partial_fit(data)
        except AttributeError as e:
            if str(e).find("""object has no attribute 'partial_fit'"""):
                data = pad_single_value(self.data)
                self.model.fit(data)
            else:
                raise

    def extend(self, other):
        if self.n_columns != other.n_columns:
            raise Exception('self.n_columns != other.n_columns')
        self.history.extend(other.history)
        self.append(other.data)

    def plot(self, ax, hist_color='grey', model_color='b'):
        """Plots comparison of data histogram and model.

        Args:
          * ax (matplotlib.Axes)
        """
        def color_yticklabels(axes, color):
            for tl in axes.get_yticklabels():
                tl.set_color(color)

        # Plot histogram of data
        n, bins, patches = ax.hist(self.data, label='data histogram', 
                                   color=hist_color)
        color_yticklabels(ax, hist_color)
        ax.set_ylabel('count', color=hist_color)
        
        # Plot model fit
        model_ax = ax.twinx()
        model_x = np.linspace(bins[0], bins[-1], 100)
        logprob = self.model.score(model_x)
        model_line, = model_ax.plot(model_x, np.exp(logprob), label='model fit',
                                    color=model_color, linewidth=3)
        color_yticklabels(model_ax, model_color)
        model_ax.set_ylabel('probability density', color=model_color)

        # Legend
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        model_handles, model_labels = model_ax.get_legend_handles_labels()
        ax.legend(ax_handles + model_handles, ax_labels + model_labels)

        # Make space for legend.
        def extend_ylim(axes):
            ylim = axes.get_ylim()
            axes.set_ylim([ylim[0], ylim[1]*1.2])
        extend_ylim(ax)
        extend_ylim(model_ax)

        # Title
        ax.set_title('Comparison of data histogram and model')


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
