import numpy as np
import pylab as pl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def plot_steady_states(ax, states, offset=0, 
                       color='g', label='Steady state'):
    """
    Args:
      * ax (Axes)
      * states (pd.DataFrame): Steady States
      * offset (int or float): add this to ss.mean
      * color (str)
      * label (str)

    Returns:
      * line
    """
    line = None
    for start, val in states.iterrows():
        end = val['end']
        try:
            mean = val['power'].mean + offset
        except AttributeError:
            mean = val['power'].get_model().mean + offset
        line, = ax.plot([start, end], [mean, mean], color=color, 
                     linewidth=2, alpha=0.6)
    if line is not None:
        line.set_label(label)
    return line


def plot_spike_histogram(ax, spike_histogram, bin_edges, 
                         title='Spike histogram'):
    """
    Args:
      * ax (maplotlib.Axes)
      * spike_histogram (pd.DataFrame): returned by spike_histogram()
      * bin_edges (np.ndarray, ndim=1, int): returned by spike_histogram()
      * title (str): Plot title.
    """
    n_bins = spike_histogram.columns.size
    vmax = np.max(spike_histogram.values)
    img = np.transpose(spike_histogram.values.astype(int))
    num_start_time = mdates.date2num(spike_histogram.index[0])
    num_end_time = mdates.date2num(spike_histogram.index[-1])
    ax.imshow(img, aspect='auto', vmin=0, vmax=vmax, 
               interpolation='none', origin='lower',
               extent=(num_start_time, num_end_time, 0, n_bins))
    ax.set_ylabel('Bin edges in watts')
    ax.set_title(title)
    ax.set_yticks(np.arange(n_bins+1))
    ax.set_yticklabels(['{:.0f}'.format(w) for w in bin_edges])
    return ax


def plot_clusters(ax, db, X, title_append='', scale_x=1):
    labels = db.labels_
    core_samples = db.core_sample_indices_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    if scale_x == 1:
        X_copy = X
    else:
        X_copy = X.copy()
        X_copy[:,0] *= scale_x

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # black used for noise
            col = 'k'
            markersize = 6
        class_members = [index[0] for index in np.argwhere(labels == k)]
        for index in class_members:
            x = X_copy[index]
            if index in core_samples and k != -1:
                markersize = 6
            else:
                markersize = 3
            # ax.plot(x[0], x[1], 'o', markerfacecolor=col, 
            #         markeredgecolor='k', markersize=markersize)    
            ax.plot(x[0], x[1], '.', color=col, markersize=markersize)

    ax.set_title('clustering using DBSCAN, eps={}, min_samples={}, {}'
                  .format(db.eps, db.min_samples, title_append))
    ax.set_ylabel('count')
    ax.set_xlabel('date time')


def plot_multiple_linear_regressions(ax, mlr, window_size=10, ax2=None):
    """
    Args:
        mlr: output from the multiple_linear_regressions() function
        window_size (int): Width of each windows in number of samples.  
            Must be multiple of 2.  Windows are overlapping:
              2222
            11113333

    """
    n_windows = mlr.shape[0]
    for i in range(n_windows-1):
        (slope, intercept, r_squared, std_err) = mlr.iloc[i]
        X = [mlr.index[i], mlr.index[i+1]]
        ax.plot(X, [intercept, intercept+(slope*window_size)], color='r')
        if ax2:
            ax2.plot(X, [std_err, std_err], color='k')

    ax.set_title('Multiple linear regressions')
    ax.set_ylabel('watts')

    if ax2:
        ax2.set_title('std_err')


def plot_appliance_hypotheses(ax, appliances):
    """
    Args:
      * ax (matplotlib.Axes)
      * appliances (pd.DataFrame): output from Disaggregator.disaggregate()
    """

    unique_appliances = {}
    for start, row in appliances.iterrows():
        app = row['appliance']
        if not unique_appliances.has_key(app):
            unique_appliances[app] = []
        unique_appliances[app].append([start, row['end']])
    n_unique_appliances = len(unique_appliances)

    # create list of colors
    colors = plt.cm.jet(np.linspace(0,1,n_unique_appliances))

    # now iterate through unique_appliances
    y = 0
    handles = []
    labels = []
    for app, data in unique_appliances.items():
        for x in data:
            line, = ax.plot(x, [y, y], color=colors[y], linewidth=5)
        handles.append(line)
        labels.append(app.label)
        y += 1

    ax.legend(handles, labels)
    

def plot_data_and_model(data, model, ax=None, 
                        hist_color='grey', model_color='b'):

    if ax is None:
        ax = plt.gca()

    def color_yticklabels(axes, color):
        for tl in axes.get_yticklabels():
            tl.set_color(color)

    # Plot histogram of data
    n, bins, patches = ax.hist(data, label='data histogram', 
                               color=hist_color, edgecolor=hist_color,
                               bins=1000)
    color_yticklabels(ax, hist_color)
    ax.set_ylabel('count', color=hist_color)

    # Plot model fit
    model_ax = ax.twinx()
    model_x = np.linspace(bins[0], bins[-1], 10000)
    logprob = model.score(model_x)
    model_line, = model_ax.plot(model_x, np.exp(logprob), label='model fit',
                                color=model_color, linewidth=1)
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
    
    return ax
