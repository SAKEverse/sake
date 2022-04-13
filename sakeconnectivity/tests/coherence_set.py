# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:22:13 2022

@author: bstone04
"""

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
from scipy import signal
import pandas as pd

#Hilbert transform to determine the amplitude envelope and 
#instantaneous frequency of an amplitude-modulated signal
from scipy.signal import butter
from scipy.signal import filtfilt

#Import plotting tools
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import seaborn as sns


def create_coh_waves(coh, length=30, fs=1000, freq=5, noise=.3,
                     phase_shift=0):
    
    phase_shift = phase_shift%360

    # get interever inteval in samples
    iei = int(fs/freq)
    
    # get phase and amplitude
    phase = np.linspace(-np.pi, np.pi, iei)
    amp = np.sin(phase)
    
    # create template and shifted waves
    orig_length = length
    trim = 1 *fs
    length = orig_length*fs + 2*trim
    template = np.zeros(length)
    shifted = np.zeros(length)
    
    # get wave index
    idx_tempalte = np.arange(0, template.shape[0]-trim, iei).astype(int)

    # get phase shift
    phase_shift_wave = np.linspace(0, 360, iei)
    const_phase_shift = np.argmin(np.abs(phase_shift_wave-phase_shift))
    m = iei*coh     # mean (max should be cycle length)
    sd = m/5        # standard deviation
    var_shift =  np.random.normal(m, sd, len(idx_tempalte)).astype(int)
    idx_shifted = idx_tempalte + var_shift +const_phase_shift
    
    # convolve waves
    template[idx_tempalte] = 0*np.random.rand(len(idx_tempalte)) + np.ones(len(idx_tempalte))
    shifted[idx_shifted] = 0*np.random.rand(len(idx_tempalte)) + np.ones(len(idx_tempalte))
    noise_arrayt = np.random.rand(len(template))*noise
    noise_arrays = np.random.rand(len(template))*noise
    template = np.convolve(template, amp, mode='same') + noise_arrayt
    shifted = np.convolve(shifted, amp, mode='same') + noise_arrays
    return template[trim:-trim], shifted[trim:-trim]

#define sine wave generator
def sinewave(seconds,fs,frequency1,frequency2,amp1,amp2,shift):
    """
    This function takes in time and speed parameters to generate
    two sine waves to provide a means to test coherence
    INPUT:      seconds (INT, length of signal in seconds)
                fs (INT, sampling frequency in milliseconds)
                frequency (INT, frequency of signal)
                amp1 (INT, amplitude of the first signal)
                amp2 (INT, amplitude of the second signal)
                shift (FLOAT, value to shift the second signal by)
    OUTPUT:     sinewave (array, signal of first sine wave) 
                shifted (array, signal of second sine wave)
                time (array, time domain)
    """
    
    time = np.arange(0, seconds, 1/fs)
    sinewave = amp1 * np.sin(2 * np.pi * frequency1 * time)
    shifted = amp2 * np.sin(2 * np.pi/shift * frequency2 * time)
    return sinewave, shifted, time


def psd_calc(signal1, signal2, fs):
    """
    This function takes in two signals and speed parameters to compute the power
    (and cross power) spectral density estimates
    INPUT:      signal1 (array, first region's signal)
                signal2 (array, second region's signal)
                fs (INT, sampling frequency in milliseconds)
                overlap (INT, number of points to overlap between signal)
    OUTPUT:     f (array, sample frequencies) 
                Pxx (array, power spectral density of signal1)
                Pyy (array, power spectral density of signal2)
                Pxy (array, cross power spectral density of signal1, signal2)
    """
    
    # calculate cross power spectral density
    fx, Pxy = signal.csd(signal1, signal2, fs = fs, nperseg=fs, noverlap = int(fs/2))
    
    # compute power spectral density estimates using Welch's method 
    psds = []                       #generate empty list for storage
    for sig in [signal1,signal2]:
        f, psd = signal.welch(sig, fs= fs, nperseg=fs, noverlap = int(fs/2))
        psds.append(psd)
    
    Pxx, Pyy = psds
    
    return f, Pxx, Pyy, Pxy


#define spectral coherence measures
def spectral_coherence(Pxx, Pyy, Pxy, mode = 'mne'):
    """
    This function takes in two signals and speed parameters to compute the power
    (and cross power) spectral density estimates
    INPUT:      Pxx (array, power spectral density of signal1)
                Pyy (array, power spectral density of signal2)
                Pxy (array, cross power spectral density of signal1, signal2)
                mode (string ['mne' or 'scipy'], defaults to 'mne')
    OUTPUT:     Cxy (array, square root magnitude [mne] magnitude squared [scipy]
                     coherence of two signals) 
    """
    if mode == 'mne':
        Cxy = np.abs(Pxy) / np.sqrt(Pxx * Pyy)
    
    if mode == 'scipy':
        Cxy = np.abs(Pxy)**2 / (Pxx * Pyy)
    
    return Cxy

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

def extracthilbertphase(signal1, signal2, fs, freqs):
    
    #construct dataframe for band extraction
    df = pd.concat([pd.DataFrame(freqs, columns =['band', 'low', 'high'])]*2)  
    df.insert(0,'signal', value = np.repeat(['signal1','signal2'],len(freqs)))
    df.insert(4,'value', value = [item for sublist in [[signal1]*len(freqs), [signal2]*len(freqs)] for item in sublist]) 
    
    filtsigs = []; phases = []
    for signal_in in df.signal.unique():
        for band in df.band.unique():
            dset = df.loc[(df['signal']==signal_in)& (df['band']==band)]
            out = butter_bandpass_filter(data = dset['value'].iloc[0], lowcut = dset['low'].iloc[0], 
                        highcut = dset['high'].iloc[0], fs = fs, order=2)
            filtsigs.append(out)
            phases.append(np.angle(signal.hilbert(out),deg=False))
            
    #store back in dataframe
    df.insert(5,'filtered', value = filtsigs)
    df.insert(6,'phase', value = phases)
    
    return df

def phase_locking_value(df, freqs):
    """
    This function takes in two signals and speed parameters to compute the power
    (and cross power) spectral density estimates
    INPUT:      Pxy (array, cross power spectral density of signal1, signal2)
    OUTPUT:     PLV (array, phase-locking value by frequency) 
    """
    plvs = []
    for band in df.band.unique():
        dset = df.loc[df.band==band]
        complex_phase_diff = np.exp(complex(0,1)*(dset.phase.iloc[0] - dset.phase.iloc[1]))
        PLV = np.abs(np.sum(complex_phase_diff))/len(dset.phase.iloc[0])
        plvs.append(PLV)
        
    df = pd.DataFrame(freqs, columns =['band', 'low', 'high'])
    df.insert(3,'PLV', value = plvs)
    return df
    
#generate a sine wave and a shifted one
fs = 1000
seconds = 5
frequency_1 = 20
frequency_2 = 20
amplitude_1 = 2
amplitude_2 = 5
shift = 1
# cleanwave, shifted, time = sinewave(seconds, fs, frequency_1, frequency_2, \
#                                     amplitude_1, amplitude_2, shift)
    
cleanwave, shifted = create_coh_waves(0.2, length=seconds, fs=fs, freq=20, noise=.8,
                      phase_shift=180)

#specify 'standard' frequency/bands
iter_freqs = [('lowtheta', 3, 6),('hightheta', 6, 12),\
              ('beta', 13, 30),('gamma', 30, 80)]
    
# =============================================================================
# =============================================================================
# #                             #SIGNAL GENERATION
# =============================================================================
# =============================================================================
#apply some noise to the waves to simulate signals
# noisewave = cleanwave+np.random.normal(scale=1, size=cleanwave.size)
# noise_shiftwave = shifted+np.random.normal(scale=1, size=shifted.size)

# =============================================================================
# =============================================================================
# #                               #CALCULATIONS
# =============================================================================
# =============================================================================

# calculate welch estimate on signal to verify power matches specified frequencies
#f, Pxx_den = signal.welch(noisewave, fs= fs, noverlap=250)

# set signal variables for testing functions
signal1 = cleanwave
signal2 = shifted

# caculate psds
f, Pxx, Pyy, Pxy = psd_calc(signal1, signal2, fs)

# calculate standard coherence
Cxy = spectral_coherence(Pxx, Pyy, Pxy, mode = 'mne')

# filter signals and extract instantaneous phases
lfp_df = extracthilbertphase(signal1, signal2, fs, iter_freqs)

# calculate Phase-Locking Value (PLV)
plv_df = phase_locking_value(lfp_df, iter_freqs)


a = lfp_df[lfp_df['band']=='beta']
phase1 = a.iloc[0]['phase']
phase2 = a.iloc[1]['phase']
h = np.histogram(phase1 - phase2, bins=100)
pdiff = h[1][np.argmax(h[0])]
# plt.plot(phase1-phase2)
plt.hist(phase1-phase2,bins=100)

# # =============================================================================
# # =============================================================================
# # #                                 #PLOTTING
# # =============================================================================
# # =============================================================================
# # set up global subplot grid
# fig = plt.figure(constrained_layout=True,figsize=(12,7))
# gs = GridSpec(2, 3, figure=fig)
# ax1 = fig.add_subplot(gs[0, :]); ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[1:, 1]); ax4 = fig.add_subplot(gs[1, 2])

# # plot signals
# # ax1.plot(signal1, color='midnightblue',lw=1,alpha=0.3)
# # ax1.plot(signal2, color='seagreen',lw=1,alpha=0.5)
# ax1.plot(signal1,color='midnightblue',lw=3, label='Region 1')
# ax1.plot(signal2,color='seagreen',lw=3, label='Region 2')
# ax1.set_title('Raw Signals',fontweight='bold')
# ax1.legend(ncol=2, loc= 3)

# # plot psds
# f, Pxx_den = signal.welch(signal1,fs= fs, nperseg=fs, noverlap = int(fs/2))
# ax2.semilogy(f, Pxx_den,color='midnightblue',lw = 3, label='Region 1: %iHz' %(frequency_1))
# f, Pxx_den = signal.welch(signal2,fs= fs, nperseg=fs, noverlap = int(fs/2))
# ax2.semilogy(f, Pxx_den,color='seagreen',lw = 3, label='Region 2: %iHz' %(frequency_2))
# ax2.set_xlim(0,100); ax2.set_xlabel('frequency [Hz]'); ax2.set_ylabel('PSD [V**2/Hz]')
# ax2.legend(); ax2.set_title('PSDs',fontweight='bold')
# ax2.set_ylim(10**-3.5,10^1)
# ax2.set_ylabel('PSD $V^2$/Hz'); ax2.set_xlabel('Frequency (Hz)')

# #plot spectral coherence
# ax3.plot(f, Cxy, color='lightseagreen',lw=3); ax3.set_xlim(0,100)
# ax3.set_title('Spectral Coherence',fontweight='bold')
# ax3.set_ylim(0, 1.1)
# ax3.set_ylabel('CSD $V^2$/Hz'); ax3.set_xlabel('Frequency (Hz)')

# #plot Phase Locking Value (PLV)
# colorset = sns.color_palette("Greens",len(plv_df))
# g = sns.barplot(x = 'band', y = 'PLV', ax = ax4,\
#                 palette = colorset, data = plv_df)
# ax4.set_xlabel('Band'); ax4.set_ylim(0,1.1)
# ax4.set_title('Phase-Locking Value',fontweight='bold')

    
    
    
    