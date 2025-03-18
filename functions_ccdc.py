import scipy.stats as stats
import scipy as sp 
import time as time
from sklearn.mixture import GaussianMixture
import os
import rasterio
from rasterio.plot import show
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
import warnings
import time 
from scipy.signal import lombscargle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
#%matplotlib inline
#plt.style.use('default')
import scipy
from scipy.interpolate import interp1d
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
#from skimage.measure import block_reduce
from functions_current import *



#med_use_lim is quantile value below which the median error value is used 
def get_errors(data, error_type='diff', med_use_lim=0.5):
    """
    Computes error estimates for a given time series.

    Parameters:
    - data (array-like): 1D NumPy array of time-series values.
    - error_type (str): 'diff' for difference-based errors, 'const' for constant errors.
    - med_use_lim (float): Quantile threshold for replacing small errors with the median.

    Returns:
    - errors (numpy array): Array of error estimates.
    """
    data = np.asarray(data)  # Ensure it's a NumPy array
    
    if len(data) < 2:
        raise ValueError("Data must contain at least two elements to compute errors.")

    diff = np.diff(data)  # Compute first-order differences
    
    if error_type == 'diff':
        errors = np.zeros(len(data))
        
        # Compute mean absolute difference for interior points
        errors[1:-1] = (np.abs(diff[:-1]) + np.abs(diff[1:])) / 2
        
        # Set first and last values separately
        errors[0] = np.abs(diff[0])
        errors[-1] = np.abs(diff[-1])

        # Apply median thresholding to limit error values
        threshold = np.quantile(errors, min(max(med_use_lim, 0), 1))  # Keep within valid range
        median_error = np.median(errors)
        errors = np.where(errors > threshold, errors, median_error)

    elif error_type == 'const':
        errors = np.full(len(data), np.mean(data) / 50)  # Constant error estimate

    else:
        raise ValueError("Invalid error_type. Choose 'diff' or 'const'.")
    
    return errors

def harmonic1(x,a,b,c,d):
    return a+b*x+c*np.cos(2*np.pi*x)+d*np.sin(2*np.pi*x)

def harmonic2(x,a,b,c,d,e,f):
    return a+b*x+c*np.cos(2*np.pi*x)+d*np.sin(2*np.pi*x)+e*np.cos(4*np.pi*x)+f*np.sin(4*np.pi*x)

def harmonic3(x,a,b,c,d,e,f,g,h):
    return a+b*x+c*np.cos(2*np.pi*x)+d*np.sin(2*np.pi*x)+e*np.cos(4*np.pi*x)+f*np.sin(4*np.pi*x)+g*np.cos(6*np.pi*x)+h*np.sin(6*np.pi*x)


def get_line(start_time, stop_time, par, model='harmonic1', nn=200):
    """
    Generates a time series line using a specified harmonic model.

    Parameters:
    - start_time (float): Start of the time range.
    - stop_time (float): End of the time range.
    - par (list or array): Parameters for the selected harmonic model.
    - model (str): Harmonic model type ('harmonic1', 'harmonic2', 'harmonic3').
    - nn (int): Number of points to generate.

    Returns:
    - lin (numpy array): 2D array with shape (2, nn). 
    - Row 0: time values, Row 1: model values.
    """
    lin = np.zeros((2, nn))
    lin[0, :] = np.linspace(start_time, stop_time, nn)  # Vectorized time generation
    
    # Apply the selected harmonic model
    if model == 'harmonic1':
        if len(par) != 4:
            raise ValueError("harmonic1 requires 4 parameters.")
        lin[1, :] = harmonic1(lin[0, :], *par)
    
    elif model == 'harmonic2':
        if len(par) != 6:
            raise ValueError("harmonic2 requires 6 parameters.")
        lin[1, :] = harmonic2(lin[0, :], *par)
    
    elif model == 'harmonic3':
        if len(par) != 8:
            raise ValueError("harmonic3 requires 8 parameters.")
        lin[1, :] = harmonic3(lin[0, :], *par)
    
    else:
        raise ValueError(f"Invalid model '{model}'. Choose 'harmonic1', 'harmonic2', or 'harmonic3'.")
    
    return lin

import numpy as np
import scipy.optimize

def fit_function(data, errors, time, est_yes=False, function='harmonic1', weighted=False, est=None, print_yes=True):
    """
    Fits a harmonic function to the given data using nonlinear least squares.

    Parameters:
    - data (array-like): The observed data points.
    - errors (array-like): The measurement uncertainties (used if `weighted=True`).
    - time (array-like): The time points corresponding to `data`.
    - est_yes (bool): If False, default initial estimates are used.
    - function (str): The harmonic function to fit ('harmonic1', 'harmonic2', 'harmonic3').
    - weighted (bool): If True, uses `errors` as weights.
    - est (array-like): Initial parameter estimates (if None, defaults are used).
    - print_yes (bool): If True, prints the fitted parameters.

    Returns:
    - res (numpy array): Fitted parameters of the chosen harmonic function.
    """
    # Define default estimates if not provided
    default_estimates = {
        'harmonic1': np.array([5000., -2.5, 170., 180.]),
        'harmonic2': np.array([5000., -2.5, 170., 180., 20.2, 20.2]),
        'harmonic3': np.array([5000., -2.5, 170., 180., 19., 19., 3., 3.])
    }
    
    # Select function dynamically
    harmonic_models = {
        'harmonic1': harmonic1,
        'harmonic2': harmonic2,
        'harmonic3': harmonic3
    }

    # Validate function name
    if function not in harmonic_models:
        raise ValueError(f"Invalid function '{function}'. Choose from 'harmonic1', 'harmonic2', or 'harmonic3'.")

    # Set initial estimate if `est_yes` is False
    if not est_yes:
        est = default_estimates[function]
    
    # Ensure `est` is a NumPy array
    est = np.array(est)

    # Fit the function
    fit_args = (harmonic_models[function], time, data, est)
    if weighted:
        res, cov = scipy.optimize.curve_fit(*fit_args, sigma=errors, absolute_sigma=True)
    else:
        res, cov = scipy.optimize.curve_fit(*fit_args)

    if print_yes:
        print(f"Fitted parameters for {function}: {res}")

    return res

#is run on an element, can be bigger time line or single time element
#excludes snow, shadow and shadow, haze is kept in standard model
import numpy as np

def get_met_mask(meta, met_limit=0.5, haze=False):
    """
    Generates a boolean mask based on meteorological data constraints.

    Parameters:
    - meta (numpy array): 1D or 2D meteorological data array.
    - met_limit (float): Threshold value for filtering.
    - haze (bool): If True, applies additional haze condition.

    Returns:
    - sel (numpy array): Boolean mask indicating where conditions are met.
    """
    meta = np.asarray(meta)  # Ensure input is a NumPy array

    # Define indices for filtering based on haze condition
    idx_list = [1, 2, 5] if not haze else [1, 2, 3, 5]

    # Check if `meta` has enough dimensions
    if meta.ndim == 2:
        if meta.shape[0] <= max(idx_list):
            raise ValueError(f"meta array has insufficient rows ({meta.shape[0]}) for requested indices {idx_list}")
        sel = np.all(meta[idx_list, :] < met_limit, axis=0)  # Vectorized check

    elif meta.ndim == 1:
        if meta.shape[0] <= max(idx_list):
            raise ValueError(f"meta array has insufficient elements ({meta.shape[0]}) for requested indices {idx_list}")
        sel = np.all(meta[idx_list] < met_limit)  # Vectorized check for 1D array

    else:
        raise ValueError(f"Unsupported meta dimension: {meta.ndim}. Expected 1D or 2D.")

    return sel


def init_fit(data, dates, element, mask, meta, channel=0, clean_lim=3, function='harmonic1', 
            met_limit=0.5, haze=False, lim_1_7=5, lim_2_8=5):
    """
    Fits a harmonic model to time-series data, applying data cleaning and filtering.

    Parameters:
    - data (numpy array): 3D array (bands x time x elements) of reflectance data.
    - dates (numpy array): 1D array of time points (years).
    - element (int): Index of the spatial element to process.
    - mask (numpy array): Boolean mask of valid data points.
    - meta (numpy array): 3D meteorological data (features x time x elements).
    - channel (int): Index of the spectral band to use.
    - clean_lim (float): Standard deviation threshold for filtering.
    - function (str): Model type ('harmonic1', 'harmonic2', 'harmonic3').
    - met_limit (float): Meteorological filtering threshold.
    - haze (bool): Whether to apply haze filtering.
    - lim_1_7, lim_2_8 (float): Limits for `clean_data`.

    Returns:
    - res3 (numpy array): Fitted parameters for the final model.
    - st3 (float): Standard deviation of the final residuals.
    """
    # Select one year of data
    t_max = np.argmin(np.abs(dates - dates[0] - 1)) + 1
    sel_dat = data[:, :t_max, element]
    sel_metdat = meta[:, :t_max, element]
    sel_time = dates[:t_max]

    # Clean data using provided limits
    ndata2, ndates2, x = clean_data(sel_dat, sel_time, lim_1_7=lim_1_7, lim_2_8=lim_2_8)

    # Apply meteorological mask
    x2 = get_met_mask(sel_metdat, met_limit=met_limit, haze=haze)

    # Combine masks
    x3 = x & x2

    # Filter valid data
    ndata = sel_dat[:, x3]
    ndates = sel_time[x3]

    # Initial fit (unweighted)
    res1 = fit_function(ndata[channel, :], ndates, ndates, weighted=False, function=function)

    # Compute errors
    errors = get_errors(ndata[channel, :], med_use_lim=0.9)

    # Refined fit (weighted)
    res2 = fit_function(ndata[channel, :], errors, ndates, weighted=True, est_yes=True, est=res1, function=function)

    # Prepare arrays for model predictions
    linb = np.zeros((3, len(sel_time)))
    linc = np.zeros((3, len(ndates)))

    # Fit predictions using the selected function
    harmonic_models = {'harmonic1': harmonic1, 'harmonic2': harmonic2, 'harmonic3': harmonic3}
    model_func = harmonic_models.get(function)
    
    if model_func is None:
        raise ValueError(f"Invalid function '{function}'. Choose from 'harmonic1', 'harmonic2', or 'harmonic3'.")

    linb[0] = model_func(sel_time, *res1)
    linb[1] = model_func(sel_time, *res2)
    linc[0] = model_func(ndates, *res1)
    linc[1] = model_func(ndates, *res2)

    # Compute residual statistics
    print(np.median(errors))
    st1 = np.std(linc[0] - ndata[channel, :])
    st2 = np.std(linc[1] - ndata[channel, :])
    med1 = scipy.stats.median_abs_deviation(linc[0] - ndata[channel, :]) * 1.483
    med2 = scipy.stats.median_abs_deviation(linc[1] - ndata[channel, :]) * 1.483
    print(st1, st2, med1, med2)

    # Apply selection mask for additional cleaning
    selection = (np.abs(linc[0] - ndata[channel, :]) / st1 < clean_lim) | \
                (np.abs(linc[1] - ndata[channel, :]) / st2 < clean_lim)

    ndata2 = ndata[:, selection]
    ndates2 = ndates[selection]

    print(ndata.shape, ndata2.shape)

    # Final model fitting
    res3 = fit_function(ndata2[channel, :], errors, ndates2, weighted=False, est_yes=True, est=res1, function=function)

    # Generate final predictions
    linb[2] = model_func(sel_time, *res3)
    lind = np.zeros((3, len(ndates2)))
    lind[0] = model_func(ndates2, *res3)

    # Compute final residual statistics
    st3 = np.std(lind[0] - ndata2[channel, :])
    med3 = scipy.stats.median_abs_deviation(lind[0] - ndata2[channel, :]) * 1.483
    print(st3, med3)

    # Plot results
    plt.plot(sel_time, sel_dat[channel, :], 'o', ms=4, color='blue', label=f'Ch {channel+1}')
    plt.errorbar(ndates, ndata[channel, :], yerr=errors, fmt='o', ms=2, color='red', label=f'Ch {channel+1} cleaned a')
    plt.plot(ndates2, ndata2[channel, :], '+', ms=10, color='black', label=f'Ch {channel+1} cleaned b')
    plt.plot(sel_time, linb[0], '-', color='cornflowerblue', label='Fit 1')
    plt.plot(sel_time, linb[1], '-', color='green', label='Fit 2')
    plt.plot(sel_time, linb[2], '-', color='red', label='Fit 3')
    plt.title(f"Segment {element}")
    plt.xlabel("Time [year]")
    plt.ylabel("Surface Reflectance")
    plt.legend()
    plt.show()

    return res3, st3

def comb_truth_lists(sel1, sel2):
    """
    Combines two Boolean lists, replacing `True` values in `sel1` with corresponding values from `sel2`.

    Parameters:
    - sel1 (list of bool): Primary Boolean list.
    - sel2 (list of bool): Secondary Boolean list, used to replace `True` values in `sel1`.

    Returns:
    - list: Combined Boolean list of the same length as `sel1`.
    """
    sel2_combined = []
    c = 0  # Counter for sel2 index
    
    
    for i in range(len(sel1)):
        if not sel1[i]:  # If sel1[i] is False, retain False
            sel2_combined.append(False)
        else:
            if c < len(sel2):  # Ensure we do not exceed sel2 length
                sel2_combined.append(sel2[c])
                c += 1
            else:
                raise ValueError("Mismatch: `sel2` has fewer elements than expected based on `sel1`.")

    return sel2_combined

def get_model_pred(times, par, function='harmonic1'):
    """
    Generates predictions from a specified harmonic model.

    Parameters:
    - times (array-like): Time values for which predictions are needed.
    - par (array-like): Parameters for the selected harmonic model.
    - function (str): Name of the harmonic function ('harmonic1', 'harmonic2', 'harmonic3').

    Returns:
    - numpy array: Model predictions.
    """
    # Define available harmonic functions
    harmonic_models = {
        'harmonic1': harmonic1,
        'harmonic2': harmonic2,
        'harmonic3': harmonic3
    }

    # Validate function selection
    if function not in harmonic_models:
        raise ValueError(f"Invalid function '{function}'. Choose from 'harmonic1', 'harmonic2', or 'harmonic3'.")

    # Validate parameter length
    expected_params = {'harmonic1': 4, 'harmonic2': 6, 'harmonic3': 8}
    if len(par) != expected_params[function]:
        raise ValueError(f"Incorrect number of parameters for {function}. Expected {expected_params[function]}, got {len(par)}.")

    # Compute and return model predictions
    return harmonic_models[function](times, *par)


#all bands at ones for now only rmse mode
#mask not needed here because all tested 

def test_one_new(model, newdata, newtime, st, max_chan=8, fact=3, print_yes=True, function='harmonic1', forward=True):
    """
    Tests whether a new data sample fits within a threshold of the harmonic model.

    Parameters:
    - model (numpy array): Model parameters (shape: [num_params, max_chan]).
    - newdata (numpy array): Observed data values for different channels.
    - newtime (array-like): Time points corresponding to `newdata`.
    - st (numpy array): Standard deviation estimates per channel.
    - max_chan (int): Maximum number of spectral bands/channels.
    - fact (float): Scaling factor for thresholding.
    - print_yes (bool): Whether to print the computed score.
    - function (str): Harmonic function to use ('harmonic1', 'harmonic2', 'harmonic3').
    - forward (bool): Placeholder argument (not currently used).

    Returns:
    - bool: `True` if the data is within threshold, `False` otherwise.
    """
    model = np.asarray(model)
    newdata = np.asarray(newdata)
    newtime = np.asarray(newtime)
    st = np.asarray(st)
    
    # Ensure correct dimensions for `model`
    if model.shape[1] != max_chan:
        raise ValueError(f"Expected `model` with {max_chan} channels, but got shape {model.shape}.")

    # Generate model predictions for all channels
    predictions = np.array([get_model_pred(newtime, model[:, i], function=function) for i in range(max_chan)])

    # Compute normalized errors (avoiding division by zero)
    error_components = np.abs(predictions - newdata) / (fact * (st + 1e-8) * max_chan)

    # Sum across channels
    total_error = np.sum(error_components)

    if print_yes:
        print("Total Error:", total_error)

    return total_error <= 1
            

def init_fit_all_chan(data, dates, element, meta, clean_lim=2.5, inter_per=1.15, fin_per=1, init_per=1.3, 
                        lim_1_7=5, lim_2_8=5, start_frame=0, max_chan=8, bad_chan=3, print_yes=True, 
                        plot_yes=True, function='harmonic1', met_limit=0.5, haze=False, forward=True, 
                        med_use_lim=0.9):
    """
    Multi-channel harmonic model fitting with progressive filtering.

    Parameters:
    - data (numpy array): Shape (bands x time x elements) - Reflectance data.
    - dates (numpy array): Time points (years).
    - element (int): Index of spatial element.
    - meta (numpy array): Shape (meta_features x time x elements) - Meteorological data.
    - clean_lim (float): Standard deviation threshold for filtering.
    - inter_per (float): Intermediate period (years).
    - fin_per (float): Final period (years).
    - init_per (float): Initial period (years).
    - start_frame (int): Starting index for analysis.
    - max_chan (int): Maximum number of spectral bands.
    - bad_chan (int): Allowed number of bad channels before rejection.
    - print_yes (bool): Print intermediate statistics.
    - plot_yes (bool): Whether to generate plots.
    - function (str): Harmonic model ('harmonic1', 'harmonic2', 'harmonic3').
    - met_limit (float): Threshold for meteorological filtering.
    - haze (bool): Whether to apply haze filtering.
    - forward (bool): Whether to analyze data forward in time.
    - med_use_lim (float): Threshold for median filtering.

    Returns:
    - ares3 (numpy array): Final fitted harmonic model parameters.
    - st3 (numpy array): Standard deviations of final residuals.
    - good_dat (numpy array): Cleaned dataset.
    - full_selection (numpy array): Boolean mask of selected data.
    """
    # Select initial time window
    if forward:
        t_max = np.argmin(np.abs(dates - dates[start_frame] - init_per)) + 1
        sel_dat = data[:, start_frame:t_max, element]
        sel_metdat = meta[:, start_frame:t_max, element]
        sel_time = dates[start_frame:t_max]
    else:
        stop_frame = -1 - start_frame
        start_frame = np.argmin(np.abs(dates - dates[stop_frame] + init_per)) + 1
        sel_dat = data[:, start_frame:stop_frame, element]
        sel_metdat = meta[:, start_frame:stop_frame, element]
        sel_time = dates[start_frame:stop_frame]

    # Apply cleaning filters
    ndata2, ndates2, selection1a = clean_data(sel_dat, sel_time, lim_1_7=lim_1_7, lim_2_8=lim_2_8)
    selection1b = get_met_mask(sel_metdat, met_limit=met_limit, haze=haze)
    selection1c = selection1a & selection1b

    # Select valid data
    ndata_prelim = sel_dat[:, selection1c]
    ndates_prelim = sel_time[selection1c]

    # Select the intermediate period
    if forward:
        t_max = np.argmin(np.abs(ndates_prelim - ndates_prelim[0] - inter_per)) + 1
        ndata, ndates, selection1 = ndata_prelim[:, :t_max], ndates_prelim[:t_max], selection1c[:t_max]
    else:
        stop_frame = -1 - start_frame
        start_frame = np.argmin(np.abs(ndates_prelim - ndates_prelim[stop_frame] + inter_per)) + 1
        ndata, ndates, selection1 = ndata_prelim[:, start_frame:stop_frame], ndates_prelim[start_frame:stop_frame], selection1c[start_frame:stop_frame]

    # Initialize model parameter storage
    param_size = {'harmonic1': 4, 'harmonic2': 6, 'harmonic3': 8}[function]
    ares1, ares2, ares3 = np.zeros((param_size, max_chan)), np.zeros((param_size, max_chan)), np.zeros((param_size, max_chan))

    # Initial model fitting
    for i in range(max_chan):
        ares1[:, i] = fit_function(ndata[i, :], ndates, ndates, weighted=False, print_yes=False, function=function)
        errors = get_errors(ndata[i, :], med_use_lim=med_use_lim)
        ares2[:, i] = fit_function(ndata[i, :], errors, ndates, weighted=True, est_yes=True, est=ares1[:, i], print_yes=False, function=function)

    # Model predictions
    linb = np.array([get_model_pred(sel_time, ares1[:, i], function=function) for i in range(max_chan)])
    linc = np.array([get_model_pred(ndates, ares1[:, i], function=function) for i in range(max_chan)])

    # Outlier detection
    st1, st2 = np.std(linc - ndata, axis=1), np.std(linc - ndata, axis=1)
    sels = np.abs(linc - ndata) / (st1[:, None] + 1e-8) < clean_lim
    selection = np.mean(sels, axis=0) > 1 - bad_chan / max_chan

    # Ensure selection is not shorter than `selection1c`
    selection = np.pad(selection, (0, len(selection1c) - len(selection)), constant_values=False)
    full_selection2 = comb_truth_lists(selection1c, selection)
    ndata3, ndates3 = sel_dat[:, full_selection2], sel_time[full_selection2]

    # Select final period
    t_max = np.argmin(np.abs(ndates3 - ndates3[0] - fin_per)) + 1
    ndata2, ndates2, full_selection = ndata3[:, :t_max], ndates3[:t_max], full_selection2[:t_max]

    # Final model fitting
    good_dat = np.zeros((9, len(ndates2)))
    good_dat[8] = ndates2
    for i in range(max_chan):
        ares3[:, i] = fit_function(ndata2[i, :], get_errors(ndata2[i, :], med_use_lim=med_use_lim), ndates2, weighted=False, est_yes=True, est=ares1[:, i], print_yes=False, function=function)
        good_dat[i] = ndata2[i, :]

    # Compute final residuals
    st3 = np.std(np.array([get_model_pred(ndates2, ares3[:, i], function=function) for i in range(max_chan)]) - ndata2, axis=1)

    if print_yes:
        print(st3)

    return ares3, st3, good_dat, full_selection
    
    
#mask not needed is not executed when not good
import numpy as np

def fit_one_more(data_time, plusdata, plustime, est, max_chan=8, print_yes=False, function='harmonic1', forward=True):
    """
    Extends a harmonic model fit with additional data points.

    Parameters:
    - data_time (numpy array): Existing data matrix (shape: [9, time_steps]).
    - plusdata (numpy array): Additional data values (shape: [max_chan, new_time_steps]).
    - plustime (numpy array): Corresponding time values for `plusdata`.
    - est (numpy array): Initial model parameter estimates (shape: [num_params, max_chan]).
    - max_chan (int): Number of spectral channels.
    - print_yes (bool): Whether to print the computed errors.
    - function (str): Harmonic function ('harmonic1', 'harmonic2', 'harmonic3').
    - forward (bool): Whether to process data forward or backward.

    Returns:
    - ares3 (numpy array): Updated harmonic model parameters.
    - st3 (numpy array): Standard deviation of residuals.
    - good_dat (numpy array): Updated dataset with the new observations.
    """
    # Combine existing and new time values
    all_t = np.concatenate((data_time[8, :], np.atleast_1d(plustime)))
    # Stack spectral data correctly
    all_val = np.vstack((data_time[:8, :].T, plusdata))

    # Initialize model parameter storage
    param_size = {'harmonic1': 4, 'harmonic2': 6, 'harmonic3': 8}[function]
    ares3 = np.zeros((param_size, max_chan))

    # Fit the harmonic model for each channel
    for i in range(max_chan):
        ares3[:, i] = fit_function(all_val[:, i], all_t, all_t, weighted=False, 
                                    est_yes=True, est=est[:, i], print_yes=print_yes, function=function)

    # Generate model predictions in a vectorized way
    lind = np.array([get_model_pred(all_t, ares3[:, i], function=function) for i in range(max_chan)])

    # Compute standard deviation of residuals
    st3 = np.std(lind - all_val.T, axis=1)

    if print_yes:
        print("Residual standard deviations:", st3)

    # Append new time points to the dataset
    good_dat = np.vstack((all_val.T, all_t))

    # If backward processing, shift the latest entry to the first column
    if not forward:
        good_dat = np.hstack((good_dat[:, -1][:, np.newaxis], good_dat[:, :-1]))

    return ares3, st3, good_dat


#meta mask and snow mask needs now used here including setting the optional arguments for it


def fit_after_init(data,time,first_est,first_std,good_dat,truth_vec,meta,snow,element=0,init_per=1.3,clean_lim=2.5,max_chan=8,max_fit=4,fact=3,needed_anomalies=4,print_yes=True,function='harmonic1',met_limit=0.5,haze=False,forward=True,bad_chan=4,fin_per=1.0,inter_per=1.15):
    if forward==True:
            counter=int(len(truth_vec))    
    else:
            counter=data.shape[1]-int(len(truth_vec))-2
    max_sh=time.shape[0]
    count_an=0
    found_anomaly=0
    if function=='harmonic1':
        a_est=np.zeros((4,max_chan,max_fit))
    elif function=='harmonic2':
        a_est=np.zeros((6,max_chan,max_fit))    
    elif function=='harmonic3':
        a_est=np.zeros((8,max_chan,max_fit))   
    #range of the different segments
    a_range=np.zeros((2,max_fit))     
    #do range here that it cannot be omitting when fitrst n are anomalous
    a_range[0,0]=np.min(good_dat[8])
    a_range[1,0]=np.max(good_dat[8])    
    a_est[:,:,0]=first_est
    a_std=np.zeros((8,max_fit))
    a_std[:,0]=first_std
    delt_time=init_per
    #here the question is wehther that can be reversed in time without redoing it  
    while counter<max_sh and count_an<needed_anomalies and delt_time>=init_per and counter>=0:
        #check whether bad meta data, the ignored  
        if print_yes==True:
            print(a_range,found_anomaly)                
        #meta mask
        result0b=get_met_mask(meta[:,counter,element],met_limit=met_limit,haze=haze)
        #combine the two masks 
        result0=(snow[counter]==True) & (result0b==True)
        if print_yes==True:
            print(f"flags are {result0}")
        if result0==False:  
            truth_vec.append(result0)
            result=False
        else:
            #test signifinance of it 
            result=test_one_new(a_est[:,:,found_anomaly],data[:,counter,element],time[counter],a_std[:,found_anomaly],max_chan=max_chan,fact=fact,print_yes=print_yes,function=function,forward=forward)
            if print_yes==True:
                print(f"value good is {result}")
            truth_vec.append(result)
            if result==False:
                count_an+=1
        if result==True and result0==True:
            if found_anomaly==0:
                a_est[:,:,found_anomaly],a_std[:,found_anomaly],good_dat=fit_one_more(good_dat,data[:,counter,element],time[counter],a_est[:,:,found_anomaly],print_yes=False,function=function,forward=forward)
                a_range[0,found_anomaly]=good_dat[8,0]
            else:
                a_est[:,:,found_anomaly],a_std[:,found_anomaly],good_dat2=fit_one_more(good_dat2,data[:,counter,element],time[counter],a_est[:,:,found_anomaly],print_yes=False,function=function,forward=forward)
                a_range[0,found_anomaly]=good_dat2[8,0]  
            if forward==True:    
                a_range[1,found_anomaly]=time[counter]
            if print_yes==True:
                print("fit done")
            count_an=0
        if count_an==needed_anomalies:
            found_anomaly+=1
            delt_time=np.abs(time[-1]-time[counter-needed_anomalies+1])
            if print_yes:
                print(f"found_anomaly {found_anomaly}")
                print(f"time remaining {delt_time} years")
            if delt_time>init_per:
                a_est[:,:,found_anomaly],a_std[:,found_anomaly], good_dat2,truth_vec2=init_fit_all_chan(data,time,element,meta,start_frame=counter-needed_anomalies+1,clean_lim=clean_lim,bad_chan=bad_chan,print_yes=False,plot_yes=False,function=function,forward=forward,fin_per=fin_per,inter_per=inter_per,init_per=init_per)
                #insert also range here that it cannot be nothing when a fit happens
                a_range[0,found_anomaly]=np.min(good_dat2[8])
                a_range[1,found_anomaly]=np.max(good_dat2[8])    
                #reset anomaly counter
                truth_vec=truth_vec[:-needed_anomalies]
                truth_vec=truth_vec+truth_vec2
                good_dat=np.append(good_dat,good_dat2,axis=1)
                count_an=0
                #set counter to one year after new start
                counter=np.argmin(np.abs(time-time[counter-needed_anomalies+1]-1))
                if print_yes==True:
                    print(f"new counter is at {counter}")
            else:
                #add false to truth_vec that length is right
                while len(truth_vec)<data.shape[1]:
                    truth_vec.append(False)
                return a_est,a_std,found_anomaly, truth_vec, a_range
        if forward==True:    
            counter+=1
        else:
            counter-=1
        if print_yes==True:
            print(f"counter is {counter}")
        #return
    if found_anomaly==0:
        while len(truth_vec)<data.shape[1]:
            truth_vec.append(False)
        return a_est,a_std, found_anomaly, truth_vec, a_range
    else:  
        while len(truth_vec)<data.shape[1]:
            truth_vec.append(False)        
        return a_est,a_std, found_anomaly, truth_vec, a_range    


def ccdc_v1(rdata,dates,element,meta,lim_1_7=2,lim_2_8=2,met_limit=0.5,haze=False,
                init_per=1.3,inter_per=1.15,fin_per=1.0,med_use_lim=0.9,
                function='harmonic1',forward=True,clean_lim=2.5,bad_chan=4,
                max_chan=8,max_fit=4,fact=3,needed_anomalies=4,print_yes=False):
    """
    Continuous Change Detection and Classification (CCDC) implementation using harmonic regression.

    Parameters:
    - rdata (numpy array): 3D array (bands x time x elements) of reflectance data.
    - dates (numpy array): 1D array of float dates (fraction of year).
    - element (int): Index of the element (e.g., pixel or group) to process.
    - meta (numpy array): 3D array (meta_features x time x elements) of auxiliary metadata.
    - lim_1_7 (float): Snow detection threshold (Band 1 / Band 7 ratio).
    - lim_2_8 (float): Snow detection threshold (Band 2 / Band 8 ratio).
    - met_limit (float): Filtering threshold for metadata quality control.
    - haze (bool): Whether to consider haze in metadata filtering.
    - init_per (float): Initial time window for model fitting (years).
    - inter_per (float): Intermediate time window for model refinement (years).
    - fin_per (float): Final time window for fitting (years).
    - med_use_lim (float): Quantile threshold for uncertainty estimation.
    - function (str): Harmonic function to use ('harmonic1', 'harmonic2', 'harmonic3').
    - forward (bool): Whether to fit forward in time (backward not fully supported).
    - clean_lim (float): Standard deviation threshold for outlier detection.
    - bad_chan (int): Number of bands required to classify an epoch as anomalous.
    - max_chan (int): Maximum number of spectral bands (default: 8).
    - max_fit (int): Maximum number of independent fits attempted.
    - fact (float): Standard deviation factor for detecting anomalies.
    - needed_anomalies (int): Consecutive anomalies required to classify a change.
    - print_yes (bool): Whether to print debugging information.

    Returns:
    - fit_ex3 (numpy array): Estimated harmonic model parameters after final fit.
    - st_ex3 (numpy array): Standard deviations of residuals from final fit.
    - an_ex3 (int): Number of anomalies detected.
    - truth_vec3 (list): Boolean vector tracking detected anomalies.
    - range3 (numpy array): Time ranges associated with different fits.
    """
    
    snow=get_snow2(rdata[:,:,element],type_return='bool',lim_1_7=lim_1_7,lim_2_8=lim_2_8)
    fin_fit,fin_std, good_dat,truth_vec=init_fit_all_chan(rdata,dates,element,meta,init_per=init_per,med_use_lim=med_use_lim,clean_lim=clean_lim,bad_chan=bad_chan,print_yes=False,plot_yes=False,function=function,met_limit=met_limit,haze=haze,forward=forward,lim_1_7=lim_1_7,lim_2_8=lim_2_8,fin_per=fin_per,inter_per=inter_per)
    fit_ex3,st_ex3,an_ex3,truth_vec3,range3=fit_after_init(rdata,dates,fin_fit,fin_std,good_dat,truth_vec,meta,snow,init_per=init_per,element=element,needed_anomalies=needed_anomalies,clean_lim=clean_lim,max_chan=max_chan,max_fit=max_fit,fact=fact,bad_chan=bad_chan,print_yes=print_yes,function=function,met_limit=met_limit,haze=haze,forward=forward,fin_per=fin_per,inter_per=inter_per)
        
    return fit_ex3,st_ex3,an_ex3,truth_vec3,range3


                    

#now relative to median to central hal


def get_snow2(data, lim_1_7=2, lim_2_8=2, type_return='result'):
    """
    Identifies snow-covered pixels using Band 1/7 and Band 2/8 ratios.
    
    Parameters:
    - data (numpy array): 2D array of shape (bands, time_steps) containing reflectance values.
    - lim_1_7 (float): Threshold for Band 1/7 ratio.
    - lim_2_8 (float): Threshold for Band 2/8 ratio.
    - type_return (str): 'result' returns binary mask (1=snow, 0=no snow),
                        'bool' returns boolean mask (True=not snow, False=snow).
    
    Returns:
    - numpy array: Binary (0,1) or Boolean mask depending on `type_return`.
    """

    # Ensure `data` has the correct shape
    assert data.ndim == 2, f"Expected a 2D array (bands, time_steps), but got shape {data.shape}"

    # Avoid division by zero (adding a small epsilon)
    rat17 = data[0, :] / (data[6, :] + 1e-8)
    rat28 = data[1, :] / (data[7, :] + 1e-8)

    # Compute quartiles for robust filtering
    med1s = np.quantile(rat17, [0.25, 0.5, 0.75])
    med2s = np.quantile(rat28, [0.25, 0.5, 0.75])
    
    med1, med2 = med1s[1], med2s[1]
    medq1, medq2 = (med1s[2] - med1s[0]) / 2, (med2s[2] - med2s[0]) / 2

    # Identify snow pixels
    snow = np.where((np.abs(rat17 - med1) / medq1 > lim_1_7) & 
                    (np.abs(rat28 - med2) / medq2 > lim_2_8), 1, 0)

    # Handle output format
    if type_return == 'result':
        return snow
    elif type_return == 'bool':
        return snow == 0  # True = Not Snow, False = Snow
    else:
        raise ValueError(f"Invalid `type_return`: {type_return}. Expected 'result' or 'bool'.")
    
    
#needs 8 band data (could be changed later) of one segement/pixel
def clean_data(data,dates,satellite='superdove',lim_1_7=3,lim_2_8=3):
    snow=get_snow2(data,lim_1_7=lim_1_7,lim_2_8=lim_2_8)
    ndata=data[:,(snow==0)]
    ndates=dates[(snow==0)]
    return ndata,ndates, (snow==0) 



            
#for plotting ccdc results
#is also used after fitting not all parameters before useful 


def plot_data_fit(dates, rdata, element, truth_vec, fits, ranges, model='harmonic1', source='one', flags=False):
    """
    Plots CCDC results, including observed reflectance data and harmonic model fits.

    Parameters:
    - dates (numpy array): 1D array of time points (years).
    - rdata (numpy array): 3D array (bands x time x elements) of reflectance data.
    - element (int): Index of the element (e.g., pixel or region) to plot.
    - truth_vec (numpy array): Boolean array indicating anomaly-free observations.
    - fits (numpy array): Harmonic model parameters.
    - ranges (numpy array): 2D array specifying time intervals for model fits.
    - model (str): Harmonic model type ('harmonic1', 'harmonic2', 'harmonic3', or 'none' to disable fits).
    - source (str): Whether data is from a 'one' or 'many' source.
    - flags (bool): If True, highlights points identified as valid (`truth_vec`).

    Returns:
    - None (displays a plot).
    """

    colors = ['navy', 'blue', 'turquoise', 'green', 'yellow', 'red', 'magenta', 'gray']
    truth_vec = np.asarray(truth_vec)  # Convert to NumPy array
    
    # Ensure element is an integer
    element = int(element)

    plt.figure(figsize=(10, 6))

    # Loop over spectral bands
    for i in range(min(8, rdata.shape[0])):  # Ensure we do not exceed available bands
        if rdata.shape[1] <= element:
            print(f"Skipping band {i}: element index {element} is out of bounds for rdata.shape[1] = {rdata.shape[1]}")
            continue  # Skip if element is out of range

        y_values = rdata[i, :, element].flatten()  # Ensure 1D array
        if len(dates) != len(y_values):
            print(f"Skipping band {i}: Shape mismatch between dates ({dates.shape}) and reflectance ({y_values.shape})")
            continue  # Avoid plotting if the shapes do not match

        plt.plot(dates, y_values, '.', ms=1, color=colors[i], label=f'Band {i+1}' if i == 0 else "")

    # Configure plot
    plt.title(f"Segment {element}")
    plt.xlabel("Time (Year)")
    plt.ylabel("Surface Reflectance")
    plt.legend()
    plt.show()
    
#convert data frane into 2dtable of bands, quality bands, and coordinates data 
#input the data frame and how many time points are exclucluded at the end and the start 

def reshape_for_ccdc(df, exclude_start=0, exclude_end=0,mode='fast'):
    """
    Reshapes a DataFrame into CCDC-compatible format with spectral bands, metadata, and coordinates.

    Parameters:
    - df (pandas DataFrame): DataFrame containing spectral and metadata bands.
    - exclude_start (int): Number of time points to exclude from the start.
    - exclude_end (int): Number of time points to exclude from the end.

    Returns:
    - dates (numpy array): Filtered time array.
    - xydata (numpy array): 2D array of (x, y) coordinates for all elements.
    - rdata (numpy array): 3D array of spectral bands (bands x time x elements).
    - meta (numpy array): 3D array of metadata bands (bands x time x elements).
    """
    
    if mode=='fast':
        # Ensure DataFrame is sorted by time
        df.sort_values(by='time', inplace=True)

        # Extract unique time values (avoiding direct x=0, y=0 filtering)
        dates2 = df['time'].unique()

        # Validate `dates2` before slicing
        if dates2.shape[0] == 0:
            raise ValueError("No unique time values found in the DataFrame. Check the 'time' column.")

        # Handle time exclusions correctly
        if exclude_start == 0 and exclude_end == 0:
            dates = dates2
        elif exclude_end == 0:
            dates = dates2[exclude_start:]
        else:
            dates = dates2[exclude_start:-exclude_end]

        # Compute number of spatial elements
        num_elements = df.shape[0] // dates2.shape[0]  # Number of unique (x, y) locations

        # Validate division result
        if num_elements <= 0:
            raise ValueError("Computed `num_elements` is <= 0. Ensure DataFrame has valid (x, y) coordinates.")

        # Initialize data arrays
        rdata = np.zeros((8, dates.shape[0], num_elements))
        meta = np.zeros((8, dates.shape[0], num_elements))
        xydata = np.zeros((2, num_elements))

        # Extract unique (x, y) coordinates from the first time step
        selc = df[df['time'] == dates[0]].loc[:, ['x', 'y']]
        xydata[:, :] = selc.to_numpy().T

        # Iterate over all time steps and populate `rdata` and `meta`
        for i, date in enumerate(dates):
            sel = df[df['time'] == date].loc[:, ['coastal_blue', 'blue', 'turquoise', 'green', 'yellow',
                                            'red', 'red_edge', 'infrared']]
            selq = df[df['time'] == date].loc[:, ['clear', 'snow', 'shadow', 'haze_light',
                                            'haze_heavy', 'cloud', 'confidence', 'udm1']]

            rdata[:, i, :] = sel.to_numpy().T
            meta[:, i, :] = selq.to_numpy().T
    elif mode=='slow':
        dates=np.array(df[(df.x==0) & (df.y==0)].time)
        rdata=np.zeros((8,dates.shape[0],int(df.shape[0]/dates.shape[0])))
        meta=np.zeros((8,dates.shape[0],int(df.shape[0]/dates.shape[0])))
        xydata=np.zeros((2,int(df.shape[0]/dates.shape[0])))
        selc=df[df.time==dates[0]].loc[:,['x', 'y']]
        xydata[:,:]=selc.T

        for i in range(dates.shape[0]):
            #select of each date the image meta image ande coordinate information
            sel=df[df.time==dates[i]].loc[:,['coastal_blue', 'blue', 'turquoise', 'green', 'yellow',
        'red', 'red_edge', 'infrared']]
            selq=df[df.time==dates[i]].loc[:,['clear', 'snow', 'shadow', 'haze_light',
        'haze_heavy', 'cloud', 'confidence', 'udm1']]  
            rdata[:,i,:]=sel.T
            meta[:,i,:]=selq.T      
            
    return dates, xydata, rdata, meta


#harm number of harmonics, harmonic 1 has 4 parameters per fit, 2 more for each 
#nfits is the maximal number of fits for one segment

def ccdc_res_to_df(xydata, outliers, fits, i_end, harm=3, nfits=4, bands=8):
    """
    Converts CCDC model outputs into a structured pandas DataFrame.

    Parameters:
    - xydata (numpy array): (2, num_points) array containing (x, y) coordinates.
    - outliers (numpy array): (num_time_points, num_points) array of outlier flags.
    - fits (numpy array): Model fit parameters stored in a flattened format.
    - i_end (int): Number of elements to include in the final DataFrame.
    - harm (int): Number of harmonics used in model fitting (default: 3).
    - nfits (int): Maximum number of independent fits for a segment (default: 4).
    - bands (int): Number of spectral bands (default: 8).

    Returns:
    - df_res (pandas DataFrame): Structured DataFrame containing fit parameters and metadata.
    """
    
    df_res=pd.DataFrame(xydata.T[:i_end,:],columns=['x','y'])
    for i in range(outliers.shape[0]):
        vec='normal_t_'+str(i)
        df_res[vec]=outliers[i,:]
    fitpar=int(2+harm*2)
    for k in range(fitpar):
        for j in range(bands):
            for i in range(nfits):
                name='fitpar_'+str(k)+'_ch_'+str(j+1)+'_segm_'+str(i)
                #that kind of columns adding gets warnings, but works in practice 
                df_res[name]=fits[i+j*nfits+k*bands*nfits,:]          
    for j in range(bands):
        for i in range(nfits):
            name='std_ch_'+str(j+1)+'_segm_'+str(i)
            x=i+j*nfits+fitpar*bands*nfits
            df_res[name]=fits[x,:]   
    for i in range(nfits):
        x1=bands*nfits+fitpar*bands*nfits+i
        x2=bands*nfits+fitpar*bands*nfits+i+nfits
        name='start_segm_'+str(i)
        name2='end_segm_'+str(i)
        df_res[name]=fits[x1,:]   
        df_res[name2]=fits[x2,:] 
    df_res['n_anomalies']=fits[bands*nfits+fitpar*bands*nfits+2*nfits,:]    
    return df_res


#running ccdc on many elements

def run_ccdc_many(rdata, dates, meta, xydata, i_end=0, init_per=1.3, bad_chan=4, 
                med_use_lim=0.9, clean_lim=2.5, max_fit=4, fact=3, needed_anomalies=4, 
                print_yes=False, function='harmonic1', met_limit=0.5, haze=False, forward=True, 
                lim_1_7=2, lim_2_8=2, fin_per=1.0, inter_per=1.15):
    """
    Runs CCDC on multiple elements (pixels or regions) and stores results in a DataFrame.

    Parameters:
    - rdata (numpy array): 3D array (bands x time x elements) of reflectance data.
    - dates (numpy array): 1D array of time points (years).
    - meta (numpy array): 3D array (meta_features x time x elements) of auxiliary metadata.
    - xydata (numpy array): 2D array of (x, y) coordinates.
    - i_end (int): Number of elements to process (default: all elements).
    - init_per (float): Initial time window for model fitting (years).
    - bad_chan (int): Number of spectral bands required for an epoch to be flagged.
    - med_use_lim (float): Quantile threshold for uncertainty estimation.
    - clean_lim (float): Standard deviation threshold for outlier detection.
    - max_fit (int): Maximum number of independent fits per segment.
    - fact (float): Scaling factor for anomaly detection.
    - needed_anomalies (int): Consecutive anomalies required to classify a change.
    - print_yes (bool): Whether to print debug information.
    - function (str): Harmonic model type ('harmonic1', 'harmonic2', 'harmonic3').
    - met_limit (float): Filtering threshold for metadata quality control.
    - haze (bool): Whether to consider haze in metadata filtering.
    - forward (bool): Whether to fit forward in time.
    - lim_1_7 (float): Snow detection threshold (Band 1/7 ratio).
    - lim_2_8 (float): Snow detection threshold (Band 2/8 ratio).
    - fin_per (float): Final time window for fitting (years).
    - inter_per (float): Intermediate time window for model adjustments.

    Returns:
    - df_res (pandas DataFrame): Structured DataFrame containing CCDC results.
    """
    
    # Validate `rdata`
    if rdata.shape[2] == 0:
        raise ValueError("`rdata` contains no elements. Check input dimensions.")

    # Process all elements if `i_end=0`
    if i_end == 0:
        i_end = rdata.shape[2]

    # Number of spectral bands (hardcoded as 8)
    max_chan = 8

    # Determine harmonic model order
    order = {'harmonic1': 1, 'harmonic2': 2, 'harmonic3': 3}.get(function, 1)
    fit_par = 2 + 2 * order  # Number of parameters per fit

    # Compute total parameters for storage
    tot_par = fit_par * max_fit * max_chan + max_chan * max_fit + max_fit * 2 + 1

    # Initialize storage arrays
    all_good = np.zeros((dates.shape[0], i_end))  # Outlier tracking
    all_met_dat = np.zeros((tot_par, i_end))  # Model fit parameters

    # Process each element
    for i in range(i_end):
        print(i)
        if print_yes:
            print(f"Processing element {i+1}/{i_end}")

        # Run CCDC for current element
        fits, st, an, truth, rangeb = ccdc_v1(
            rdata, dates, i, meta, init_per=init_per, med_use_lim=med_use_lim,
            clean_lim=clean_lim, max_chan=max_chan, max_fit=max_fit, fact=fact, 
            needed_anomalies=needed_anomalies, print_yes=print_yes, function=function,
            met_limit=met_limit, haze=haze, forward=forward, lim_1_7=lim_1_7, lim_2_8=lim_2_8,
            bad_chan=bad_chan, fin_per=fin_per, inter_per=inter_per
        )

        # Flatten output into storage array
        rangec=np.reshape(rangeb,(max_fit*2),order='C')
        stb=np.reshape(st,(max_fit*max_chan),order='C')
        fitsb=np.reshape(fits,(fit_par*max_fit*max_chan),order='C')        
        comb=np.concatenate([fitsb,stb,rangec])
        all_met_dat[:,i]=np.append(comb,[an])
        
        print(len(truth))
        
        all_good[:,i]=truth 
    df_res=ccdc_res_to_df(xydata,all_good,all_met_dat,i_end=i_end,harm=order,nfits=max_fit,bands=max_chan)
    return df_res

#uses result from many fits file 


def plot_data_fit2(dates, rdata, afits, element, model='harmonic1', source='many2', 
                    flags=False, exclude_start=0, exclude_end=0, n_fits=4, channel=8, 
                    max_band_plot=8, quant=0.002, yrange_adjust=True):
    """
    Plots CCDC results, including observed reflectance data and harmonic model fits.

    Parameters:
    - dates (numpy array): 1D array of time points (years).
    - rdata (numpy array): 3D array (bands x time x elements) of reflectance data.
    - afits (pandas DataFrame): DataFrame containing CCDC fit results.
    - element (int): Index of the element (e.g., pixel or region) to plot.
    - model (str): Harmonic model type ('harmonic1', 'harmonic2', 'harmonic3', or 'none').
    - source (str): Whether data is from 'many2' source format.
    - flags (bool): If True, highlights points identified as valid (`truth_vec`).
    - exclude_start (int): Number of initial time points to exclude.
    - exclude_end (int): Number of final time points to exclude.
    - n_fits (int): Number of independent fits per segment.
    - channel (int): Number of spectral bands (default: 8).
    - max_band_plot (int): Maximum number of spectral bands to plot.
    - quant (float): Quantile range for adjusting y-axis.
    - yrange_adjust (bool): Whether to adjust y-axis limits based on quantiles.

    Returns:
    - None (displays a plot).
    """

    colors = ['navy', 'blue', 'turquoise', 'green', 'yellow', 'red', 'magenta', 'gray']

    # Select harmonic order based on model type
    harmonic_order = {'harmonic1': 1, 'harmonic2': 2, 'harmonic3': 3}.get(model, 1)
    fit_par = 2 + 2 * harmonic_order  # Number of parameters per fit

    if source == 'many2' and model != 'none':
        start_h = 3 + dates.shape[0]
        stop_h = fit_par * n_fits * channel + start_h
        start_r = stop_h + n_fits * channel
        stop_r = start_r + 2 * n_fits
        start_n = 3
        stop_n = start_h

        # Validate bounds before slicing
        if stop_r > afits.shape[1]:
            raise ValueError(f"Requested fit range [{start_r}:{stop_r}] exceeds DataFrame size {afits.shape[1]}.")

        # Extract fits and ranges
        ranges = np.array(afits.iloc[element, start_r:stop_r]).reshape((n_fits, 2), order='C')
        fits = np.array(afits.iloc[element, start_h:stop_h]).reshape((fit_par, channel, n_fits), order='C')
        truth_vec = np.array(afits.iloc[element, start_n:stop_n]).astype(bool)

        print("Ranges of fits are:")
        valid_ranges = ranges[ranges[:, 0] > 0]  # Remove zero-filled rows
        if valid_ranges.shape[0] == 0:
            print("No valid fit ranges found.")
        else:
            print(np.round(valid_ranges, 3))

    # Initialize array for valid data points
    adat = np.array([])

    # Plot observed reflectance data
    for i in range(min(max_band_plot, channel)):
        plt.plot(dates, rdata[i, :, element], '.', ms=1, color=colors[i], label=f'Band {i+1}' if i == 0 else "")

        # Highlight valid observations if `flags=True`
        if flags and truth_vec.size == rdata.shape[1]:
            plt.plot(dates[truth_vec], rdata[i, truth_vec, element], '.', ms=3, color=colors[i])
            adat = np.append(adat, rdata[i, truth_vec, element])

        # Plot harmonic model fits
        if model != 'none':
            for j in range(n_fits):
                if ranges[j, 0] > 0:
                    lin = get_line(ranges[j, 0], ranges[j, 1], fits[:, i, j], model=model)
                    plt.plot(lin[0], lin[1], color=colors[i])

    # Configure plot
    plt.title(f"Segment {element}")
    plt.xlabel("Time (Year)")
    plt.ylabel("Surface Reflectance")
    plt.xlim(dates[0] - 0.02, dates[-1] + 0.02)

    # Adjust y-axis limits
    if yrange_adjust and adat.size > 0:
        ylims_ar = np.quantile(adat, [quant, 1 - quant])
        plt.ylim(ylims_ar[0], ylims_ar[1])

    plt.show()
