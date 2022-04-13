# -*- coding: utf-8 -*-

####----------------------- IMPORTS ------------------- ######
import numpy as np
from numba import njit
import pandas as pd
####--------------------------------------------------- ######

def mean_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    '''
    Find outliers based on mean, using numpy convolve

    Parameters
    ----------
    arr : np.ndarray
    window : int
    threshold : float

    Returns
    -------
    outliers : np.ndarray

    '''
    
    #first make a threshold index:
        
    time_thresholds=np.convolve(arr, np.ones(window)/window, mode='valid')
    
    #forward fill the terminal values
    
    time_thresholds=np.append(time_thresholds,[time_thresholds[-1]]*(len(arr)-len(time_thresholds)))
    
    #compare the original array to the threshold index to find the outliers
    
    outliers = ((arr>(time_thresholds*threshold)) |  (arr<-(time_thresholds*threshold)))
    
    return outliers

def std_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    '''
    Find outliers based on standard deviation, using pandas

    Parameters
    ----------
    arr : np.ndarray
    window : int
    threshold : float

    Returns
    -------
    outliers : np.ndarray

    '''
    
    #first make a threshold index:
        
    time_thresholds=np.array(pd.Series(arr).rolling(window).std().fillna(method='bfill'))
    time_thresholds=time_thresholds[window//2:]
    time_thresholds=np.append(time_thresholds,[time_thresholds[-1]]*(len(arr)-len(time_thresholds)))
    import matplotlib as mpl
    mpl.pyplot.plot(time_thresholds)
    #compare the original array to the threshold index to find the outliers
    
    outliers = ((arr>(time_thresholds*threshold)) |  (arr<-(time_thresholds*threshold)))
    
    return outliers


@njit
def mad(arr):
    return np.median(np.abs(arr-np.median(arr)))

@njit
def rolling_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    """
    Find outliers from vector

    Parameters
    ----------
    time_vector : np.ndarray
    window : int
    threshold : float

    Returns
    -------
    outliers : np.ndarray

    """
    
    # create vector and convert to bool
    median_value = np.zeros(arr.shape)
    mad_value = np.zeros(arr.shape)
    
    # half window
    half_win = int(np.ceil(window/2))

    ## start
    # get baseline and threshold
    baseline = arr[:half_win]
    median_value[:half_win] = np.median(baseline)
    mad_value[:half_win] = mad(baseline)
    
    i = half_win # init counter
    
    ## middle  
    # get baseline and threshold
    for i in range(i, arr.shape[0] - half_win):
        # get baseline and threshold
        baseline = arr[i - half_win : i + half_win]  
        median_value[i] = np.median(baseline)
        mad_value[i] = mad(baseline)

    ## end
    # get baseline and threshold
    baseline = arr[i+1:]
    median_value[i+1:] = np.median(baseline)
    mad_value[i+1:] = mad(baseline)
        
    # compare the original array to the threshold index to find the outliers
    outliers = ((arr>(median_value + mad_value*threshold)) |  (arr<(median_value - mad_value*threshold)))

    return outliers


def fast_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    '''
    finds the outliers based on median by reshaping

    Parameters
    ----------
    arr : np.ndarray
    window : int
    threshold : float

    Returns
    -------
    outliers : np.ndarray

    '''
    
    if window > arr.shape[0]:
        window = arr.shape[0]
    # first make a threshold index:
    
    # divide the array into windows by reshaping, then find the median of each window, this makes an array of size: original_len/window
    reshaped = pd.DataFrame(arr[:(arr.shape[0]//window)*window].reshape(-1,window))
    
    # get median and mad
    medians = reshaped.median(axis=1)
    mads = reshaped.mad(axis=1)

    # expand the array back to full size by interpolaing the in between values
    smoothed_medians = np.interp(np.linspace(0, medians.shape[0], medians.shape[0]*window), np.arange(medians.shape[0]), medians)
    smoothed_mads = np.interp(np.linspace(0, mads.shape[0], mads.shape[0]*window), np.arange(mads.shape[0]), mads)
    
    #shift half a window to center the data
    smoothed_medians = np.append([smoothed_medians[0]]*(window//2),smoothed_medians[:-(window//2)])
    smoothed_mads= np.append([smoothed_mads[0]]*(window//2),smoothed_mads[:-(window//2)])
    
    # fill the end values 
    smoothed_mads = np.append(smoothed_mads,[smoothed_mads[-1]]*(arr.shape[0]-smoothed_mads.shape[0]))
    medians = np.append(smoothed_medians,[smoothed_medians[-1]]*(arr.shape[0]-smoothed_medians.shape[0]))

    # compare the original array to the threshold index to find the outliers
    outliers = ((arr>(medians + smoothed_mads*threshold)) |  (arr<(medians - smoothed_mads*threshold)))
    
    return outliers

# define function that will be executed

# remove 
def get_outliers(arr:np.ndarray, window:int, threshold:float) -> np.ndarray:
    
    # remove large outliers
    thresh = 4*np.std(arr) + np.mean(arr)
    outliers1 = arr>thresh
    new_median = np.median(arr[arr<thresh])
    arr[outliers1] = new_median
    
    outliers2 =  fast_outliers(arr, window, threshold)
    outliers = np.concatenate((outliers1.reshape(-1,1), outliers2.reshape(-1,1)), axis=1)
    return np.any(outliers, axis=1)





