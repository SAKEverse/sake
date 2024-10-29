# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import signal
##### ------------------------------------------------------------------- #####

def norm_zero_lag_xcorr(vec1, vec2):
    """
    Calculates normalized zero-lag cross correlation
    Returns a single number
    """
    auto_v1 = np.sum(vec1**2,axis=-1)
    auto_v2 = np.sum(vec2**2,axis=-1)
    xcorr = np.sum(vec1 * vec2,axis=-1)
    denom = np.sqrt(np.multiply(auto_v1,auto_v2))
    return np.divide(xcorr, denom)


def create_coh_waves(coh, length=30, fs=1000, freq=5, noise=.3,
                     phase_shift=0):
    
    # keep variables within bounds
    if coh > 1:
        coh = 1
    if noise > 1:
        noise = 1
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
    noise_array = np.random.rand(len(template))*noise
    template = np.convolve(template, amp, mode='same') + noise_array
    shifted = np.convolve(shifted, amp, mode='same') + noise_array
    return template[trim:-trim], shifted[trim:-trim]

# template, shifted = create_coh_waves(0.5, phase_shift=0 ,length=30, fs=1000, freq=10, noise=0.3)
# # plt.plot(template)
# # plt.plot(shifted)
# plt.plot(np.angle(signal.hilbert(template)) - np.angle(signal.hilbert(shifted)))

# r = norm_zero_lag_xcorr(template, shifted)
# print(r)
# # coh=[]
# # coh_value=[]
# # for i in np.arange(0,1,0.05):
    
# #     template, shifted = create_coh_waves(i, phase_shift=0 ,length=30, fs=1000, freq=10, noise=.3)
# #     phase1 = np.angle(signal.hilbert(template))
# #     phase2 = np.std(signal.hilbert(shifted))
# #     coh_value.append(np.mean(phase2-phase1))
# #     coh.append(i)
# # plt.plot(coh,coh_value)
# # # plt.plot()
# # # samples=1000
# # # coherence=1
# # # dphase = np.ones(samples)+ (coherence*np.random.rand(samples))
# # # plt.plot( np.sin(dphase))













