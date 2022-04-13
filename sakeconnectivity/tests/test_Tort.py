# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
import adi
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
# import seaborn as sns
from numba import njit
##### ------------------------------------------------------------------- #####

@njit
def get_phase_amp(phase_array, amp_array):
    """
    Get phase amplitude histogram and MI (Tort, 2010)

    Parameters
    ----------
    phase_array : array, phase of slow wave
    amp_array : array, amp of fast wave

    Returns
    -------
    phasebins : array,
    ampmeans : array,
    MI : float, modulation index (Tort 2010)

    """
    
    # create bins and init empty arrays
    phasebins = np.arange(-np.pi, np.pi, 0.1)
    bins = np.shape(phasebins)[0]-1
    ampmeans = np.zeros(bins)
    
    for k in range(bins):

        # For each phase bin get lower and upper limit
        pL = phasebins[k]
        pR = phasebins[k+1]
        indices = (phase_array>=pL) & (phase_array<pR)        # Find phases falling in this bin,
        ampmeans[k] = np.mean(amp_array[indices])             # ... compute mean amplitude,
        
        # Calculate phase amplitude coupling based on Tort,2010
        amplP = ampmeans/np.sum(ampmeans)
        amplQ = np.ones(bins)/bins
        distKL = np.sum(amplP*np.log(amplP/amplQ))
        MI = distKL/np.log(bins)
    return MI


def downsample(array, samplingrate, downsamplewidth=250):
    """
    Takes in array and downsamples (default 250ms) series
    based on the file sampling rate and downsampling width
    INPUT:      time series vector
    OUTPUT:     downsampled vector and calculated downsampling factor
    """
    sample_rate = int(samplingrate)
    ds_factor = int(sample_rate/downsamplewidth)       #set down-sampling factor (to 250ms)
    ds_s = signal.decimate(array.astype(np.float32), ds_factor, ftype='fir')
    return ds_factor,ds_s


#define bandpass filter parameters to parse out frequencies
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Define bandpass filter function to take in user selected 
    band starts/stops.
    INPUT:      vector of data, bandpass start/stops (integers)
    OUTPUT:     vector containing bandpass filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

@njit
def get_phase_amp_np(amp_phase_array):
    """
    Get phase amplitude histogram and MI (Tort, 2010)

    Parameters
    ----------
    phase_array : array, phase of slow wave
    amp_array : array, amp of fast wave

    Returns
    -------
    phasebins : array,
    ampmeans : array,
    MI : float, modulation index (Tort 2010)

    """
    #get arrays
    half=int(len(amp_phase_array)/2)
    amp_array = amp_phase_array[:half]
    phase_array = amp_phase_array[half:]
    
    # create bins and init empty arrays
    phasebins = np.arange(-np.pi, np.pi, 0.1)
    bins = np.shape(phasebins)[0]-1
    ampmeans = np.zeros(bins)
    
    for k in range(bins):
        # breakpoint()
        # For each phase bin get lower and upper limit
        pL = phasebins[k]
        pR = phasebins[k+1]
        indices = (phase_array>=pL) & (phase_array<pR)        # Find phases falling in this bin,
        ampmeans[k] = np.mean(amp_array[indices])             # ... compute mean amplitude,
        
        # Calculate phase amplitude coupling based on Tort,2010
        amplP = ampmeans/np.sum(ampmeans)
        amplQ = np.ones(bins)/bins
        distKL = np.sum(amplP*np.log(amplP/amplQ))
        MI = distKL/np.log(bins)
    return MI

class myObj:
    def __init__(self,arr):
        self.arr=arr

def get_phase_amp_magic(row):

    amp_array = row[0].arr
    phase_array = row[1].arr
    
    # create bins and init empty arrays
    phasebins = np.arange(-np.pi, np.pi, 0.1)
    ampmeans = np.zeros(np.size(phasebins)-1)
    phasemeans = np.zeros(np.size(phasebins)-1)
    
    for k in range(np.size(phasebins)-1):
        # For each phase bin get lower and upper limit
        pL = phasebins[k]
        pR = phasebins[k+1]
        indices = (phase_array>=pL) & (phase_array<pR)        # Find phases falling in this bin,
        ampmeans[k] = np.mean(amp_array[indices])             # ... compute mean amplitude,
        phasemeans[k] = np.mean([pL, pR])                     # ... save center phase.
        
        # Calculate phase amplitude coupling based on Tort,2010
        amplP = ampmeans/sum(ampmeans)
        amplQ = np.ones(np.size(phasebins)-1)/(np.size(phasebins)-1)
        distKL = sum(amplP*np.log(amplP/amplQ))
        MI = distKL/np.log(np.size(phasebins)-1)
    return MI


# =============================================================================
# =============================================================================
# #                               TESTS
# =============================================================================
# =============================================================================


fread = adi.read_file(r'C:\Users\panton01\Desktop\test_PAC\mouse1_mouse2_mouse2_mouse4.adicht')

ch_array = [[0,1], [3,4], [6,7], [9,10]]
for uid, ch in enumerate(ch_array):
    fs = fread.channels[0].fs[0]
    
    # read data
    bla = fread.channels[ch[0]].get_data(1)
    pfc = fread.channels[ch[1]].get_data(1)
    
    # downsample
    _, bla = downsample(bla, fs)
    _, pfc = downsample(pfc, fs)

    # get new fs
    fs = 250
    
    # get instant phase and amp
    phase_array = butter_bandpass_filter(bla, 6, 12, fs,)
    phase_array = np.angle(signal.hilbert(phase_array))
    
    amp_array = butter_bandpass_filter(pfc, 30, 80, fs,)
    amp_array = np.abs(signal.hilbert(amp_array))
    
    # window
    phase_array = phase_array[:120000]
    amp_array = amp_array[:120000]
    win = 30*fs
    newshape = (int(amp_array.shape[0]/win), win) 
    phase_array = phase_array.reshape(newshape)
    amp_array = amp_array.reshape(newshape)
    
    # tic=time()
    mi = np.zeros(amp_array.shape[0])
    for i in range(amp_array.shape[0]):
        mi[i]= get_phase_amp(phase_array[i,:], amp_array[i,:])
    # print(time()-tic)
    # tic=time()
    # amp_phase_array = np.concatenate((amp_array, phase_array), axis=1)
    # mi = np.apply_along_axis(get_phase_amp_np, 1, amp_phase_array)
    # print(time()-tic)
    # # tic=time()
    # # c = np.concatenate((amp_array[:,:,None], phase_array[:,:,None]), axis=2)
    # # d=np.apply_along_axis((lambda nd: myObj(nd)),1,c)
    # # mi=np.apply_along_axis(get_phase_amp_magic,1,d)
    plt.plot(mi, label=str(uid))
plt.legend()

# a = np.concatenate((amp_array, phase_array), axis=1)
# b = np.concatenate((amp_array[:,:,None], phase_array[:,:,None]), axis=2)
# # mi = np.apply_along_axis(get_phase_amp_np,1, a)

# c=np.array(list(zip(amp_array,phase_array)))

# np.vectorize(get_phase_amp_np)(c)
# np.apply_over_axes(get_phase_amp_np, b, [1,2])



















