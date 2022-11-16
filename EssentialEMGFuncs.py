import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb as rgb
import seaborn as sns
from scipy import signal
import ray
import ast
import time
import warnings

def resamp_dataframe(df, freq):
    resamp_time = np.arange(
        df['time'][0], df['time'].iloc[-1], 1/freq, dtype=np.float)
    values = np.zeros((len(resamp_time), len(df.columns)))
    for i in np.arange(len(df.columns)):
        values[:, i] = np.interp(
            resamp_time, df['time'].values, df.iloc[:, i].values)
    new_data = pd.DataFrame(values, columns=df.columns)
    return new_data


def create_envelope(df, fs, micro=True, band_pass=(40, 400), low_pass=5, order=6, make_fig=False, muscle_idx_to_plot=0, plot_time=np.arange(0, 10000), dark=True):
    if make_fig:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
        import cufflinks as cf
        cf.go_offline()  # needed to use plotly on the local server
        # darkmode: solar, space, henanigans lightmode: pearl, ggplot, white
        cf.set_config_file(world_readable=True, theme='henanigans')
        pyo.init_notebook_mode()
    # need to drop time because it does not make sense to filter time
    time = df['time']
    df2 = df.copy()
    df2.drop(labels='time', axis=1, inplace=True)
    if micro:
        df2 = df2.apply(lambda x: x*1000000)
    if make_fig:
        muscle_name = df.columns[muscle_idx_to_plot]
        fig = make_subplots(rows=4, cols=2)
        if dark:
            fig.update_layout(template='plotly_dark', title=muscle_name)
        else:
            fig.update_layout(template='plotly_white', title=muscle_name)
        fig.add_trace(go.Scatter(x=time[plot_time], y=df2[muscle_name].values[plot_time], mode='lines', name='raw'), row=1, col=1)
        fX, fY = signal.welch(df2[muscle_name],fs=fs)
        fig.add_trace(go.Scatter(x=fX, y=fY), row=1, col=2)

    b, a = signal.butter(order, band_pass, fs=fs, btype='bandpass')
    df2 = df2.apply(lambda x: signal.filtfilt(b, a, x), axis=0)
    if make_fig:
        fig.add_trace(go.Scatter(x=time[plot_time], y=df2[muscle_name].values[plot_time], mode='lines', name='band pass'), row=2, col=1)
        fX, fY = signal.welch(df2[muscle_name],fs=fs)
        fig.add_trace(go.Scatter(x=fX, y=fY), row=2, col=2)

    df2 = df2.apply(lambda x: np.abs(x - np.mean(x)), axis=0)
    if make_fig:
        fig.add_trace(go.Scatter(
            x=time[plot_time], y=df2[muscle_name].values[plot_time], mode='lines', name='rectified'), row=3, col=1)
        fX, fY = signal.welch(df2[muscle_name],fs=fs)
        fig.add_trace(go.Scatter(x=fX, y=fY), row=3, col=2)

    c, d = signal.butter(order,low_pass, fs=fs, btype='lowpass')
    df2 = df2.apply(lambda x: signal.filtfilt(c, d, x), axis=0)
    if make_fig:
        fig.add_trace(go.Scatter(
            x=time[plot_time], y=df2[muscle_name].values[plot_time], mode='lines', name='linear envelope'), row=4, col=1)
        fX, fY = signal.welch(df2[muscle_name],fs=fs)
        fig.add_trace(go.Scatter(x=fX, y=fY), row=4, col=2)

    # let's add back time
    df2['time'] = time
    if make_fig:
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1))
        fig.show()
    return df2


def create_normal_profile(df, events, target_length=2000):
    cols = [list(df.columns), np.arange(len(events))]
    normal_profile = pd.DataFrame(columns=pd.MultiIndex.from_product(cols, names=['muscle', 'epoch']))
    par_normalize = ray.remote(epoch_n_normalize)
    remote_id = {m:[] for m in list(df.columns)}
    idx = pd.IndexSlice
    for m in list(df.columns):
        # normal_profile.loc[idx[:], idx[m,:]] = epoch_n_normalize(df[m],df['time'],events,target_length)  #  for debugging
        remote_id[m] = par_normalize.remote(df[m],df['time'],events,target_length)
    for m in list(df.columns):
        normal_profile.loc[idx[:], idx[m,:]] = ray.get(remote_id[m])
    return normal_profile


def epoch_n_normalize(muscle_data, time_data, events, target_length):
    """
    For each trial, which is passed to "create_normal_profile" as a DF and is
    passed to this function as a dict, there is a specific set of epochs for
    every muscle and times. So, We can define two-level dataframes with indices
    as long as the *target_length*. The two levels of the DF columns are
    'muscle' and 'epoch'. The last muscle is the timestamp.
    """
    md = muscle_data.copy()
    td = time_data
    epoched = {i: [] for i in np.arange(len(events))}
    epoched_time = {i: [] for i in np.arange(len(events))}
    for i in np.arange(len(events)):
        epoched[i] = md[(td >=events.loc[i,'start']) & (td <= events.loc[i,'end'])].values
        epoched_time[i] = td[(td >=events.loc[i,'start']) & (td <= events.loc[i,'end'])].values

    normal_data = np.ndarray((target_length, len(epoched.keys())))
    for i in np.arange(len(epoched.keys())):
        normal_time = np.linspace(epoched_time[i][0], epoched_time[i][-1], target_length)
        normal_data[:, i] = np.interp(normal_time, epoched_time[i], epoched[i])
    return normal_data


def plot_normalized(epoched_muscle, fs, title=None, show_progress=False, downsample=True):
    """
    plots muscle epoches epoches stacked to examine the activity progress.

    ## INPUTS:
    `epoched_muscle`: dataframe, singel muscle epoches to be plotted.
    `fs`: the ORIGINAL frequency of the singal. For a faster representation, I downsample by defualt by a factor of 10.

    `title`: title of the figure, default: None.

    show progress: If True, epoch progess from first to last will follow 'ylgnbu' color order, default: False.

    downsample: If true, donwsamples data to accelarate plotting progress, default: True
    """
    df2 = epoched_muscle.copy()
    if downsample:
        df2 = df2.apply(lambda x: signal.decimate(x, 10, zero_phase=True))
    if not show_progress:
        df2.iplot(colors=['orange' for _ in np.arange(
            len(df2))], showlegend=False, title=title)
    else:
        df2.iplot(colorscale='ylgnbu', showlegend=False, title=title)


def plot_mean(muscle_df, target_muscles=None, plot_CI=True, CI_method='bootstrap', titles=None, downsample=True, plot_events = False, step_time=None, event_to_plot='pert'):
    """
    Plot average EMG activations.

    This function plots EMG mean profiles for a single subject, and if set True, it will also plot events as vertical lines. THis function does not warp the profiles to the vents. Use `warped_average` for time locked profiles.

    Paramters
    ---------
    muscle_df : DataFrame
        epoched & normalized EMG dataframe. This DF is the result of the `normalize_epoch' and has multi-level coulmns. The first level is the muscle names and the second level is the epcovh numbers.

    target_muscles : list, default None
        Which muscles from the `muscle_df` should be pltted. If `None`, all muscles will be plotted on top of each other. TODO: #2 If this is a list, only a those muscle will be plotted. If this is dictionary, each group of muscles corresponding to each key will be plotted in a subplot.

    plot_CI : bool, defualt True
        calcualtes and plots the confidence interval.

    CI_method : str, default bootstrap'
        If set to `bootstrap`, it will use bootstrapping stats to calcualte confidence intervals. Other option is `parameteric` to use the standard CI calcualtion.

    titles : str, default None
        the title of the sub/figure. If the title is a dictionary, with the same lenght as the target_muscle

    downsample : bool, default True
        Reduces the number of the datapoints for an easier and faster plotting.
    
    plot_events : bool, default False
        Wheter to plot events as vertical lines on the mean plot.
    
    step_time : DataFrame, default None
        the usual dataframe contaitng step events. Only used to calcualte the latency of the events if `plot_events = True`
    
    event_to_plot : str, default 'pert'
        The event from `step_time` to plot as a vertical line. The mean profiles are not warped though. TODO: Make it a list, so we can plot multiple events.
    
    Returns
    -------
    fig : matplotlib.pyplot.figure

    """
    if titles is not None and len(target_muscles) is not len(titles):
            warnings.warn('mismatch between title length and target_muscle length. Make sure for each target_muscle.key you have a repective name, or set title=None. Default title will be printed.')
            titles = None
    if titles is None:
        if type(target_muscles) is dict: # changing title to name of muscle-pairs
            titles = {i: j for i,j in zip(target_muscles.keys(), ['-'.join(x) for x in target_muscles.values()])}

    if type(target_muscles) is dict:
        df2 = muscle_df[[i for pairs in target_muscles.values() for i in pairs]].copy()
    elif type(target_muscles) is list:
        df2 = muscle_df[[i for i in target_muscles]].copy()
        target_muscles = {i:[t] for i,t in zip(range(len(target_muscles)),target_muscles)}
        titles = dict(zip(range(len(titles)),titles))
    elif type(target_muscles) is str:
        target_muscles = {1:target_muscles}
        df2 = muscle_df[str(target_muscles.values())].copy()
    # else:
    #     df2 = muscle_df.copy() 
    fig, axes = plt.subplots(np.int16(np.floor(len(target_muscles.keys())/2)),2,sharex=True, sharey=True,figsize=(15, 10))
    axes = axes.reshape(np.size(axes))  # this changes makes a row of axes

    if plot_events:
        event = event_to_plot
        if step_time is not None:
            avg_stride_time = np.mean(step_time['end'] - step_time['start'])
            # instead of latency, calculate the ratio.
            avg_event_rat = (np.mean(step_time[event][step_time[event] != 0] - step_time['start'][step_time[event] != 0]))/avg_stride_time
            avg_otherStep_rat = np.mean(step_time['other'] - step_time['start'])/avg_stride_time
        else:
            plot_events = False
    par_ci = ray.remote(_ci)
    emg_mean = {m:[] for m in df2.columns.get_level_values(0).unique()}
    ci_idx = {m:[] for m in df2.columns.get_level_values(0).unique()}
    for m in df2.columns.get_level_values(0).unique():
        emg_mean[m] =  df2[m].mean(axis=1)
        if plot_CI:
            ci_idx[m] = par_ci.remote(np.transpose(df2[m].to_numpy()),method=CI_method)    
    for ax, title, tm in zip(axes,titles.values(),range(len(target_muscles))):
        for m in target_muscles[tm]:
            ax.plot(emg_mean[m],label=m, linewidth=2)
            if plot_CI:
                ci_low, ci_high = ray.get(ci_idx[m])
                ax.fill_between(np.arange(len(ci_high)),np.transpose(ci_high), np.transpose(ci_low), alpha=0.6)
                if plot_events:
                    ax.axvline(np.floor(avg_event_rat*len(ci_high)), label=event, c='green')
                    ax.axvline(np.floor(avg_otherStep_rat*len(ci_high)), label='other_leg', c='blue',ls=':')
                # ax.plot(np.transpose(ci_high),label=m, linewidth=2)
                # ax.plot(np.transpose(ci_low),label=m, linewidth=2)
        ax.set_title(title)
    return fig           


def bootstrap_confidence_interval(arr, weights=[], ci=.95, n_bootstraps=2000,
                                  stat_fun='mean', random_state=None):
    """
    Get confidence intervals from non-parametric bootstrap. Adapted from MNE.
    
    Parameters
    ----------

    arr : ndarray, shape (n_samples, ...)
        The input data on which to calculate the confidence interval.
    
    ci : float
        Level of the confidence interval between 0 and 1.
    
    n_bootstraps : int
        Number of bootstraps.
    
    stat_fun : str | callable
        Can be "mean", "median", or a callable operating along ``axis=0``.
    
    random_state : int | float | array_like | None
        The seed at which to initialize the bootstrap.
    
    Returns
    -------
    cis : ndarray, shape (2, ...)
        Containing the lower boundary of the CI at ``cis[0, ...]`` and the upper boundary of the CI at ``cis[1, ...]``.
    """
    if stat_fun == "mean":
        def stat_fun(x):
            return x.mean(axis=0)
    elif stat_fun == 'median':
        def stat_fun(x):
            return np.median(x, axis=0)
    # elif not callable(stat_fun):
    #     raise ValueError("stat_fun must be 'mean', 'median' or callable.")
    n_trials = arr.shape[0]
    indices = np.arange(n_trials, dtype=int)  # BCA would be cool to have too
    rng = check_random_state(random_state)
    boot_indices = rng.choice(indices, replace=True,
                              size=(n_bootstraps, len(indices)))
    if stat_fun is not 'weighted_average':
        stat = np.array([stat_fun(arr[inds]) for inds in boot_indices])
    else:
        stat = np.array([np.average(arr[inds],weights=weights[inds],axis=0) for inds in boot_indices])
    ci = (((1 - ci) / 2) * 100, ((1 - ((1 - ci) / 2))) * 100)
    ci_low, ci_up = np.percentile(stat, ci, axis=0)
    return np.array([ci_low, ci_up])


# adapted from scikit-learn utils/validation.py
def check_random_state(seed):
    """
    Turn seed into a numpy.random.mtrand.RandomState instance.
    If seed is None, return the RandomState singleton used by np.random.mtrand.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.mtrand.RandomState(seed)
    if isinstance(seed, np.random.mtrand.RandomState):
        return seed
    try:
        # Generator is only available in numpy >= 1.17
        if isinstance(seed, np.random.Generator):
            return seed
    except AttributeError:
        pass
    raise ValueError('%r cannot be used to seed a '
                     'numpy.random.mtrand.RandomState instance' % seed)


def _parametric_ci(arr, ci=.95):
    """
    Calculate the `ci`% parametric confidence interval for `arr`. From MNE.
    """
    mean = arr.mean(0)
    if len(arr) < 2:  # can't compute standard error
        sigma = np.full_like(mean, np.nan)
        return mean, sigma
    from scipy import stats
    sigma = stats.sem(arr, 0)
    return stats.t.interval(ci, loc=mean, scale=sigma, df=arr.shape[0])


def _ci(arr, ci=.95, method="bootstrap", n_bootstraps=2000, random_state=None):
    """
    Calculate confidence interval. Aux function for plot_compare_evokeds.
    """
    if method == "bootstrap":
        return bootstrap_confidence_interval(arr, ci=ci,
                                             n_bootstraps=n_bootstraps,
                                             random_state=random_state)
    else:
        return _parametric_ci(arr, ci=ci)


def read_normal_profile(file_path):
    """
    This method imports FEATHER time-normalized EMG epoches. FEATHER file format
    changes the column names to a tuple. So, we need to re-organzie the files to
    to recreate the normal_envelope format we had in importEMG.py. 
    """
    normal_envelope = pd.read_feather(file_path)
    # column names are tuples, the first element is the muscle names and the
    # second element is the epoch number. Note that the epoch number should be
    # converted to an integer.
    mult_index_tuples = [ast.literal_eval(ind)
                         for ind in normal_envelope.columns]
    mult_index_tuples = [(m, int(n)) for m, n in mult_index_tuples]
    mult_index = pd.MultiIndex.from_tuples(mult_index_tuples)
    mult_envelope = pd.DataFrame(normal_envelope.values, columns=mult_index)
    return mult_envelope


def compute_coActivation(epoched_muscle_pair, time, step_time, event='pert', duration=400, fillnopert=True, normalize_amplitude=False, mode='traditional'):
    """
    Compute co-activation of a pair of muscles across epochs.

    This method computes co-activation of a pair of muscles for a set of epochs. Both muscles should be included in the epoched_muscle_pair in the format created by the create_normal_profile method.

    Paramters
    ---------
    epoched_muscle_pair : DataFrame
        Dataframe with two level columns. level 0 is the muscle names and level 1 is the epoch numbers (as integers). By default, Agonist is the first and Antagonist is the second in each pair.
    
    time : DataFrame
        the time-stamp for `epoched_muscle_pair`. `time` should have the same length as `epoched_muscle_pair`.
    
    step_time : DataFrame
        The usual dataframe contaitng step events.
    
    event : str
        The name of the column from `step_time` as the start of the coactivation comuptation (default `pert`).
    
    duration : int, str, default 400
        The time span starting from `event` time stamp to include in coactivation computation. If an event is passed, the window will be between the `event` and duration, for example, if `event='pert'` and `duration='other'`, then coactivation will be calucalated between *pert* and *other* events.

    normalize_amplitude : bool, default False
        normalizes the amplitdue of the muscle activation across the dataframe and then computes the coactivation (default False) 

    mode : str, default 'traditional'
        How to compute the coactivation, see Banks et.al. Front Neurol 2017 for details. Modes are `traditional`, `fixed`, and `wasted_contraction`.

    Returns
    -------
    coAcitvation : Series
        a Series with a single coAcitvation number for each epoch.
    """

    df2 = epoched_muscle_pair.copy()
    pairs_names = df2.columns.get_level_values(0).unique()
    # check if step time has values for all events, otherwise fill it with the average of the currently available values.
    if fillnopert:
        s_t = fill_perturb_time(step_time)
    else:
        s_t = step_time.copy()
    
    if normalize_amplitude: # set to False if EMG is already normalized
        for m in pairs_names:
            df2[m] = df2[m].div(df2[m].mean().mean())

    activation = pd.DataFrame(columns=pairs_names)
    coActivation = pd.Series()
    idx = pd.IndexSlice
    for e in time.columns:
        if str(duration).isnumeric():
            frames = [f for f, t in enumerate(time.loc[:, e]) if ((t >= s_t.loc[e, event]) & (t < s_t.loc[e, event]+duration/1000))]
        else:
            frames = [f for f, t in enumerate(time.loc[:, e]) if ((t >= s_t.loc[e, event]) & (t < s_t.loc[e, duration]))]
        if mode == 'fixed':
            for m in pairs_names: 
                activation.loc[idx[e], idx[m]] = np.trapz(
                    df2.loc[frames, (m, e)], time.loc[frames, e])   
            coActivation.loc[idx[e]] = 2 * activation[pairs_names[1]][e] / (
                activation[pairs_names[0]][e]+activation[pairs_names[1]][e])
        else:
            # let's make it simple
            muscle1 = df2.loc[frames, (pairs_names[0], e)].values
            muscle2 = df2.loc[frames, (pairs_names[1], e)].values
            t = time.loc[frames, e].values
            diff = np.diff(np.sign(muscle1-muscle2),append=np.sign(muscle1[-1]-muscle2[-1]))
            switch_idx = np.where(diff)[0]  # where ant/agonist switches
            if diff[0] == 0:
                switch_idx = np.append([0], switch_idx)
            switch_idx = np.append(switch_idx,[len(diff)-1])  # will add 1 later
            if mode == 'traditional':
                agonist = antagonist = 0
                for i in range(len(switch_idx[:-1])):
                    muscle1_area = np.trapz(muscle1[switch_idx[i]:switch_idx[i+1]+1],t[switch_idx[i]:switch_idx[i+1]+1])
                    muscle2_area = np.trapz(muscle2[switch_idx[i]:switch_idx[i+1]+1],t[switch_idx[i]:switch_idx[i+1]+1])
                    if muscle1_area > muscle2_area:
                        agonist += muscle1_area
                        antagonist += muscle2_area
                    else:
                        agonist += muscle2_area
                        antagonist += muscle1_area
                coActivation.loc[idx[e]] = 2*antagonist/(agonist+antagonist)
            elif mode == 'wasted_contraction':
                wasted = effective = []
                for i in range(len(switch_idx[:-1])):
                    if np.mean(muscle1[switch_idx[i]:switch_idx[i+1]+1]) > np.mean(muscle2[switch_idx[i]:switch_idx[i+1]+1]):
                        wasted = np.append(wasted,muscle2[switch_idx[i]:switch_idx[i+1]+1])
                        effective = np.append(effective, muscle1[switch_idx[i]:switch_idx[i+1]+1]-muscle2[switch_idx[i]:switch_idx[i+1]+1])
                    else:
                        wasted = np.append(wasted,muscle1[switch_idx[i]:switch_idx[i+1]+1])
                        effective = np.append(effective, muscle2[switch_idx[i]:switch_idx[i+1]+1]-muscle1[switch_idx[i]:switch_idx[i+1]+1])
                coActivation.loc[idx[e]] = np.mean(wasted)/np.max(effective)

    return coActivation


def fill_perturb_time(step_time):
    '''
    This method fills the perturb time for the unperturbed events with the average perturbation latency on the perturbed strides.
    ## Input:
    step_time: a SINGLE dataframe containing events for start and perturbations for each epoch.
    '''
    s_t = step_time.copy()
    average_pert_time = np.mean(
        s_t['pert'][s_t['pert'] != 0] - s_t['start'][s_t['pert'] != 0])
    s_t['pert'][s_t['pert'] == 0] = s_t['start'][s_t['pert'] == 0] + average_pert_time
    return s_t


def quantify_emg_metric(muscle, time, step_time, event='pert', duration=400, fillnopert=True, normalize_amplitude=False):
    """
    Compute EMG min, max, etc in a specific duration.

    This function computes an EMG metric (max, mean, etc.) of the input muscle for a set of epochs. The muscle should be in the format created by the `create_normal_profile` function.

    Parameters
    ----------
    muscle : DataFrame
        Columns are the epochs and rows are the (normalized) time span of the epoch. For example, a 2000*200 represents 200 hundred epochs, each spanning for 2000 datapoints.
        
    time : Series
        The time-stamp for `muscle`. `time` should have the same length as `muscle`.
    
    step_time : DataFrame
        The usual dataframe contaitng step events.
    
    event : str, default 'pert'
        The name of the column from `step_time` as the start of the EMG metric comuptation (default `pert`).
    
    duration: int, default 400
        The time span (in ms) starting from `event` time stamp to include in coactivation computation.
    
    fillnopert : bool, default True
        Whether to fill the pertrubation latency of strides w/o perturbations with the average perturbation latency.

    normalize_amplitude : bool, default False
        normalizes the amplitdue of the muscle activation across the dataframe and then computes the coactivation (default False)
    
    Returns
    -------
    activation : Series
        Output will be a series with the length equal to the number of epoches reporting the metric for each metric.
    """
    df2 = muscle.copy()
    
    if fillnopert:
        s_t = fill_perturb_time(step_time)
    else:
        s_t = step_time.copy()
    
    if normalize_amplitude: # set to False if EMG is already normalized
            df2 = df2.div(df2.mean().mean())
    activation = np.array([],dtype='float32')
    for e in np.arange(len(time.columns)):
        # finding the frames tha falls into the window.
        frames = [f for f, t in enumerate(time.loc[:, e]) if (
            (t >= s_t.loc[e, event]) & (t < s_t.loc[e, event]+duration/1000))]
        activation = np.append(activation,muscle.loc[frames,e].max())
    
    return activation


def warped_average(df, event_table, events_toWarp, exclude_outlier=True,  calculate_ci=False):
    """
    Warp and average strides to specifc events.

    To average multiple strides in walking or multipe trials in arm reaching, we often need to timelock the time-sereis to cetrain events rather than the begininng and end of the strides. This function calls the `time_warp` fucntion and then average over the strides.

    Parameters
    ----------
    df : DataFrame
        Time-normalized strides, can be the output of `create_normal_profile`. Rows are time frames and columns are the strides. If columns have two-level multi-index, the averge is going to be over the inner-most index. There should be `time` column in df.
    
    event_table : DataFrame
        The latency of each event for each stride. The row are the strides and the columns are the the events.
    
    events_toWarp : set
        The name of the event(s) in `event_table` that the time series should be locked to them. TODO: #3 make this a list, instead of str, we can have multiple warping points.
    
    exclude_outlier : bool, default True
        Whether to exclude strides that include outlier `events_toWarp`. Outliers are determined as +/- 5 std of the average latency for each event.
    
    calculate_ci : bool, default False
        Whether to include the ci.
    Returns
    -------
    average : DataFrame
        The wapred average of the strides.
    
    ci : DataFrame
        The lower and upper bound of the 95% confidence interval.
    """
    # idx = pd.IndexSlice
    warped_df, event_rat, other_rat = warp_toEvent(df.copy(), event_table, events_toWarp)
    average = warped_df.mean(axis=1,level=0, skipna=True)  # level=0 means to average along level 0 (i.e., muscles), so it will collaps the strides.
    if calculate_ci:
        ci_col = pd.MultiIndex.from_product([warped_df.columns.get_level_values(0).unique(),['low','high']],names=['muscle','conf_int'])
        ci_bounds = pd.DataFrame(columns=ci_col)
        for m in warped_df.columns.get_level_values(0).unique():
            ci_low, ci_high = _ci(np.transpose(warped_df[m].to_numpy()),method='bootstrap',n_bootstraps=1000)
            ci_bounds[m,'low'] = ci_low
            ci_bounds[m,'high']= ci_high
        # if any(ci_bounds.isna()):
        #     print('There is discrepancy between CI length and the average length,
        # check the code before using the results.')
        return average, event_rat, other_rat, warped_df, ci_bounds
    else:
        return average, event_rat, other_rat, warped_df


def warp_toEvent(df, event_table, events_toWarp, exclude_outlier=True):
    # TODO: #2 include outlier analysis
    # Calculate the ratio for the cents to warp to
    avg_stride_time = np.mean(event_table['end'] - event_table['start'])
    # Instead of latency, calculate the ratio.
    avg_event_rat = (np.mean(event_table[events_toWarp][event_table[events_toWarp] != 0] - event_table['start'][event_table[events_toWarp] != 0]))/avg_stride_time
    avg_otherStep_rat = np.mean(event_table['other'] - event_table['start'])/avg_stride_time

    # Now, we need to resample the strides based on the time-locking events.
    stride_frameLength = len(df)
    event_frame = np.int(np.floor(stride_frameLength*avg_event_rat))
    otherStep_frame = np.int(np.floor(stride_frameLength*avg_otherStep_rat))
    time = df['time']
    for st in time.columns:
        eventFrame_inStride = np.argmin(np.abs(time[st]-event_table[events_toWarp][st]))
        if eventFrame_inStride == 0:
            # In some cases the start frame and event frame are the same, this is because of low temporal resolution of the stepper.
            eventFrame_inStride = 1
        otherStep_inStride = np.argmin(np.abs(time[st]-event_table['other'][st]))
        # construct the warped time frame
        try:
            wTime_start_toEvent = np.linspace(time[st][0],time[st][eventFrame_inStride],event_frame, endpoint=False)
            wTime_event_toOther = np.linspace(time[st][eventFrame_inStride],time[st][otherStep_inStride], otherStep_frame-event_frame, endpoint=False)
            wTime_otherStep = np.linspace(time[st][otherStep_inStride],time[st].iloc[-1],len(time)-otherStep_frame)
            # now construct the warped dataframe, including rewriting of time column
            for m in df.columns.get_level_values(0).unique():
                w_start_toEvent = np.interp(wTime_start_toEvent,time[st][:eventFrame_inStride],df[m][st][:eventFrame_inStride])
                w_event_toOther = np.interp(wTime_event_toOther,time[st][eventFrame_inStride:otherStep_inStride],df[m][st][eventFrame_inStride:otherStep_inStride])
                w_otherStep = np.interp(wTime_otherStep,time[st][otherStep_inStride:],df[m][st][otherStep_inStride:])
                df.loc[:,(m,st)]=np.concatenate((w_start_toEvent,w_event_toOther,w_otherStep))
        except:
            print(f'Stride No {st} has a problem in events')  
    return df, avg_event_rat, avg_otherStep_rat


def twoSample_comparison_SMART(d1,d2,time_vect, t_col='time', d_col='data', sig_level=0.05, kernel_size=0.05, n_perms=1000, method='paired'):
    import SMART_Funcs as SF
    if method == 'independent':
        no_subjs1, no_subjs2 = len(d1), len(d2)
    elif method == 'paired':
        # we need to assume that the two groups have the same number of subjects
        no_subjs1 = no_subjs2 = len(d1)

    sm_data1, sm_weights1 = [np.zeros((no_subjs1,len(time_vect))) for _ in range(2)]
    sm_data2, sm_weights2 = [np.zeros((no_subjs2,len(time_vect))) for _ in range(2)]
    pm_data1, pm_weights1= [np.zeros((no_subjs1,len(time_vect),n_perms)) for _ in range(2)]
    pm_data2, pm_weights2= [np.zeros((no_subjs2,len(time_vect),n_perms)) for _ in range(2)]

    for ii in range(len(d1)):
        sm_data1[ii,:], sm_weights1[ii,:] = SF.gaussSmooth(d1[t_col][ii],d1[d_col][ii],time_vect,kernel_size)
        pm_data1[ii,:,:], pm_weights1[ii,:,:],_,_ = SF.permute(d1[t_col][ii],d1[d_col][ii],newX=time_vect,sigma=kernel_size, nPerms=n_perms, baseline=0)
    for ii in range(len(d2)):
        sm_data2[ii,:], sm_weights2[ii,:] = SF.gaussSmooth(d2[t_col][ii],d2[d_col][ii],time_vect,kernel_size)
        pm_data2[ii,:,:], pm_weights2[ii,:,:],_,_= SF.permute(d2[t_col][ii],d2[d_col][ii],newX=time_vect,sigma=kernel_size, nPerms=n_perms, baseline=0)
        
    weightedAv_data1 = np.average(sm_data1, weights = sm_weights1, axis=0)
    weightedAv_data2 = np.average(sm_data2, weights = sm_weights2, axis=0)

    sig_cL, sum_tvals = SF.clusterStat_rel(sm_data1, sm_data2, sm_weights1, sm_weights2, sig_level, method)
    
    # Calculate permutation distributions and significance thresholds
    perm_distr = SF.permuteClusterStat(pm_data1, pm_data2, pm_weights1, pm_weights2, sig_level, method)
    # perm_distr = []  # for the sake of debugging
    
    # Get significant cluster size threshold
    sig_thres = np.percentile(perm_distr, 100-(sig_level*100))
    # sig_thres = []  # for the sake of debugging
    
    # Calculate 95 confidence intervals
    conf95 = SF.weighPairedConf95(sm_data1, sm_data2, sm_weights1, sm_weights2, method)
    # conf95 = []  # Tried the non-paired CI, the paired option is more robust
    # conf95.append(bootstrap_confidence_interval(sm_data1, weights=sm_weights1,stat_fun='weighted_average'))
    # conf95.append(bootstrap_confidence_interval(sm_data2, weights=sm_weights2,stat_fun='weighted_average'))
    
    return weightedAv_data1, weightedAv_data2, sig_cL, sum_tvals, perm_distr, sig_thres, conf95


def gaussian_smoothing_df(df, time_df, kernel, num_dp):
    import SMART_Funcs as SF
    df_out = pd.DataFrame(columns=df.columns)
    for i in df_out.columns:
        df_out[i],_ = SF.gaussSmooth(time_df[i].values, df[i].values, np.linspace(time_df[i].values[0],time_df[i].values[-1],num_dp),kernel)
    
    return df_out


def compute_ampMetric(epoched_muscle, time, step_time, event='pert', duration=400, fillnopert=True, normalize_amplitude=False, mode=np.mean):
    """
    Compute co-activation of a pair of muscles across epochs.

    This method computes co-activation of a pair of muscles for a set of epochs. Both muscles should be included in the epoched_muscle_pair in the format created by the create_normal_profile method.

    Paramters
    ---------
    epoched_muscle : DataFrame
        Dataframe with one level columns. Columns are the epochs.
    
    time : DataFrame
        the time-stamp for `epoched_muscle`. `time` should have the same length as `epoched_muscle`.
    
    step_time : DataFrame
        The usual dataframe contaitng step events.
    
    event : str
        The name of the column from `step_time` as the start of the coactivation comuptation (default `pert`).
    
    duration : int, str, default 400
        The time span starting from `event` time stamp to include in coactivation computation. If an event is passed, the window will be between the `event` and duration, for example, if `event='pert'` and `duration='other'`, then coactivation will be calucalated between *pert* and *other* events.

    normalize_amplitude : bool, default False
        normalizes the amplitdue of the muscle activation across the dataframe and then computes the coactivation (default False) 

    mode : fucntion, default 'np.mean'
        What to compue for as the metric applied as an amplitude. Since this function uses Pandas `aggregate` method, you can pass aggregate-type funcations, meaning the functions that take an array and give one float number as a result.

    Returns
    -------
    ampMetric : Series
        A series with elements correspnding to each epoch.
    """

    df2 = epoched_muscle.copy()
    pairs_names = df2.columns.get_level_values(0).unique()
    # check if step time has values for all events, otherwise fill it with the average of the currently available values.
    if fillnopert:
        s_t = fill_perturb_time(step_time)
    else:
        s_t = step_time.copy()
    
    if normalize_amplitude: # set to False if EMG is already normalized
        for m in pairs_names:
            df2[m] = df2[m].div(df2[m].mean().mean())

    ampMetric = []
    idx = pd.IndexSlice
    for e in time.columns:
        if str(duration).isnumeric():
            frames = [f for f, t in enumerate(time.loc[:, e]) if ((t >= s_t.loc[e, event]) & (t < s_t.loc[e, event]+duration/1000))]
        else:
            frames = [f for f, t in enumerate(time.loc[:, e]) if ((t >= s_t.loc[e, event]) & (t < s_t.loc[e, duration]))]
        ampMetric += [df2.loc[frames,e].agg(mode)]
    return pd.Series(ampMetric)


def is_outlier(points, thresh=3):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def uncpld_bxplt(x, y, hue, data, palette='Set2', start_pos=1, ingroup_dist=1, betgroup_dist=2, ax=None, lw= 1, lc = rgb('gray'), **kwargs):
    hue_names = data[hue].unique().tolist()
    # we need to sort out some colors wrt to the hue length
    x_names = data[x].unique().tolist()
    bar_pos = start_pos
    for ii, i in enumerate(x_names):
        gstart_pos = ii*betgroup_dist + bar_pos
        for jj, j in enumerate(hue_names):
            bar_pos =  gstart_pos + jj*ingroup_dist
            data.loc[(data[x]==i)&(data[hue]==j),'t'] = bar_pos
            if jj == 0:
                data.loc[(data[x]==i)&(data[hue]==j),'tp'] = bar_pos + 0.25
            else:
                data.loc[(data[x]==i)&(data[hue]==j),'tp'] = bar_pos - 0.25
        #    data_y = data[(data[x]==i)&(data[hue]==j)][y].to_numpy()
        #    data_x = np.repeat(bar_pos,len(data_y))
    # sns.lineplot(x='t', y=y, data=data, palette=palette, alpha=.8,  lw=2, units="subj", estimator=None)   
    sns.boxplot(x=data['t'], y=data[y], palette=palette, order= np.arange(0,7).tolist(), ax=ax, **kwargs)
    sns.lineplot(x='tp', y=y, data=data, color=lc, alpha=.8,  lw=lw, units="subj", estimator=None, ax=ax)