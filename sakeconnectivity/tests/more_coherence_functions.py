# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import signal as spsignal
from test_coherence import create_coh_waves
from statsmodels.tsa.stattools import grangercausalitytests


def psd_calc(signal1, signal2, fs, windowsize, overlap):
    """
    Calucate power and cross spectra of signals.

    Parameters
    ----------
    signal1 : array, first region's signal
    signal2 : rray, second region's signal
    fs : int, sampling frequency in seconds
    overlap : int, number of points to overlap between signal

    Returns
    -------
    f : array, sample frequencies
    Pxx : array, power spectral density of signal1 
    Pyy : array, power spectral density of signal2
    Pxy : array, cross power spectral density of signal1, signal2

    """
    
    # calculate cross power spectral density
    fx, Pxy = spsignal.csd(signal1, signal2, fs=fs, nperseg=windowsize, noverlap=overlap)
    
    # compute power spectral density estimates using Welch's method 
    psds = []
    for sig in [signal1, signal2]:
        f, psd = spsignal.welch(sig, fs=fs, nperseg=windowsize, noverlap=overlap)
        psds.append(psd)
    
    Pxx, Pyy = psds    
    return f, Pxx, Pyy, Pxy


def spectral_coherence(signal1, signal2, fs, windowsize, overlap, mode='mne'):
    """
    Calulate spectral coherence using signal spectra.

    Parameters
    ----------
    signal1 : array, first region's signal
    signal2 : rray, second region's signal
    fs : int, sampling frequency in seconds
    overlap : int, number of points to overlap between signal
    mode : str, ['mne' or 'scipy'], defaults to 'mne'

    Returns
    -------
    Cxy : array, coherence value for each frequency bin.

    """
    f, Pxx, Pyy, Pxy = psd_calc(signal1, signal2, fs, windowsize, overlap)
    
    if mode == 'mne':
        Cxy = np.abs(Pxy) / np.sqrt(Pxx * Pyy)
    
    if mode == 'scipy':
        Cxy = np.abs(Pxy)**2 / (Pxx * Pyy)
    
    return f, Cxy


def imaginary_spectral_coherence(signal1, signal2, fs, windowsize, overlap, mode='mne'):
    """
    Calculate imaginary coherence using signal spectra.

    Parameters
    ----------
    signal1 : array, first region's signal
    signal2 : array, second region's signal
    fs : int, sampling frequency in seconds
    windowsize : int, length of each segment
    overlap : int, number of points to overlap between segments
    mode : str, ['mne' or 'scipy'], defaults to 'mne'

    Returns
    -------
    f : array, frequency bins
    Cxy_imag : array, imaginary coherence values for each frequency bin
    """
    # Calculate the power spectral densities and cross-spectral density
    f, Pxx, Pyy, Pxy = psd_calc(signal1, signal2, fs, windowsize, overlap)
    
    # Calculate imaginary coherence by isolating the imaginary part of the cross-spectrum
    if mode == 'mne':
        Cxy_imag = np.abs(np.imag(Pxy)) / np.sqrt(Pxx * Pyy)
    
    elif mode == 'scipy':
        Cxy_imag = (np.imag(Pxy)**2) / (Pxx * Pyy)
    
    return f, Cxy_imag


# @njit
def phase_locking_value(phase1, phase2):
    """
    Calulate phase locking value (PLV) using instantenous phase.

    Parameters
    ----------
    phase1 : array, first region's phase
    phase2 : array, second region's phase

    Returns
    -------
    plv : float, 

    """

    complex_phase_diff = np.exp(complex(0,1)*(phase1 - phase2))
    plv = np.abs(np.sum(complex_phase_diff))/phase2.shape[0]

    return plv


# @njit
def imaginary_phase_locking_value(phase1, phase2):
    """
    Calculate the imaginary part of the phase locking value (iPLV) 
    using instantaneous phase differences.

    Parameters
    ----------
    phase1 : array, first region's phase
    phase2 : array, second region's phase

    Returns
    -------
    iplv : float
        Imaginary part of the phase locking value (iPLV)
    """
    
    # Calculate the complex phase difference
    complex_phase_diff = np.exp(1j * (phase1 - phase2))
    
    # Take the imaginary part and calculate the mean of its absolute value
    iplv = np.abs(np.mean(np.imag(complex_phase_diff)))
    
    return iplv


def compute_phase(sig):
    """
    Compute the instantaneous phase of a signal using the Hilbert transform.
    
    Parameters:
        signal (np.array): The time-series data for a signal.
    
    Returns:
        phase (np.array): Instantaneous phase of the signal.
    """
    analytic_signal = spsignal.hilbert(sig)
    phase = np.angle(analytic_signal)
    return phase

def granger_causality(signal1, signal2, max_lag):
    granger_result = grangercausalitytests(np.array([signal1, signal2]).T, max_lag, verbose=False)
    f_vals = [granger_result[x][0]['ssr_ftest'][0] for x in granger_result]
    return f_vals

if __name__ == '__main__':

    # Generate example signals
    fs = 250  # Sampling frequency in Hz
    signal_1, signal_2 = create_coh_waves(coh=.8, phase_shift=60 ,length=30, fs=fs, freq=8, noise=.1)
    # np.random.shuffle(signal_1)
    # np.random.shuffle(signal_2)
    import matplotlib.pyplot as plt
    plt.plot(signal_1[:fs]);plt.plot(signal_2[:fs])
    
    # Calculate IC and PLV
    f, Cxy = spectral_coherence(signal_1, signal_2, fs=fs, windowsize=fs*5, overlap=int(fs*5/2))
    plv = phase_locking_value(compute_phase(signal_1), compute_phase(signal_2))
    # f, Cxy_imag = imaginary_spectral_coherence(signal_1, signal_2, fs=fs, windowsize=fs*5, overlap=int(fs*5/2))
    iplv = imaginary_phase_locking_value(compute_phase(signal_1), compute_phase(signal_2))
    f_vals = granger_causality(signal_1, signal_2, max_lag=np.arange(2, 20))

    print("Coherence:", np.mean(Cxy))
    # print("Imaginary Coherence:", np.mean(Cxy_imag))
    print("Phase Locking Value:", plv)
    print("Imaginary Phase Locking Value:", iplv)
    print(f_vals)

    
   

