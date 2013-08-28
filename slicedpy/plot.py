import numpy as np
import pylab as pl


def plot_steady_states(ax, states, offset=0, 
                       color='g', label='Steady state'):
    """
    Args:
        ax (Axes)
        states (pd.DataFrame): Steady States
        offset (int or float): add this to ss.mean
        color (str)
        label (str)

    Returns:
        line
    """
    line = None
    for start, val in states.iterrows():
        end = val['end']
        mean = val['power_stats'].mean + offset
        line, = ax.plot([start, end], [mean, mean], color=color, 
                     linewidth=2, alpha=0.6)
    if line is not None:
        line.set_label(label)
    return line


def plot_spike_histogram(ax, spike_histogram, bin_edges, 
                         num_start_time, num_end_time, title='Spike histogram'):
    """
    Args:
        ax (maplotlib.Axes)
        spike_histogram (np.array): returned by spike_histogram()
        bin_edges (np.array): output as second return value of spike_histogram
        num_start_time, num_end_time (float): the start and end date times for
            the time window represented by the whole spike histogram.  In
            ordinal time (where 1 == 1st Jan 0001; 2 = 2nd Jan 0001 etc)
        title (str): Plot title.
    """
    n_bins = spike_histogram.shape[0]
    vmax = spike_histogram.max()
    ax.imshow(spike_histogram, aspect='auto', vmin=0, vmax=vmax, 
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
    colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
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


def plot_multiple_linear_regressions(ax, mlr, x, window_size=10, ax2=None):
    """
    Args:
        mlr: output from the multiple_linear_regressions() function
        window_size (int): Width of each windows in number of samples.  
            Must be multiple of 2.  Windows are overlapping:
              2222
            11113333

    """

    half_window_size = window_size / 2
    n_windows = mlr.shape[0]
    start_i = 0
    end_i = window_size
    for i in range(n_windows):
        (slope, intercept, r_squared, std_err) = mlr[i]
        X = [x[start_i], x[end_i]]
        ax.plot(X, [intercept, intercept+(slope*window_size)], color='r')
        if ax2:
            ax2.plot(X, [std_err, std_err], color='k')
        start_i += half_window_size
        end_i += half_window_size
        

    ax.set_title('Multiple linear regressions')
    ax.set_ylabel('watts')

    if ax2:
        ax2.set_title('std_err')


