# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from numba import njit
from preprocess import BatchFilter
from joblib import delayed, Parallel
import multiprocessing
##### ------------------------------------------------------------------- #####

# =============================================================================
#                   Basic functions for coherence calculation
# =============================================================================

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
    fx, Pxy = signal.csd(signal1, signal2, fs=fs, nperseg=windowsize, noverlap=overlap)
    
    # compute power spectral density estimates using Welch's method 
    psds = []
    for sig in [signal1, signal2]:
        f, psd = signal.welch(sig, fs=fs, nperseg=windowsize, noverlap=overlap)
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

@njit
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

@njit
def phase_lag_index(phase1, phase2):
    """
    Calculate phase lag index (PLI).

    Parameters
    phase1 : array, first region's phase
    phase2 : array, second region's phase

    Returns
    -------
    pli : float,
    pdiff: float,
    
    The PLI ranges between 0 and 1:0 ≤ PLI ≤ 1. A PLI of zero indicates either
    no coupling or coupling with a phase difference centered around 0 mod π. A
    PLI of 1 indicates perfect phase locking at a value of Δϕ different from 0
    mod π. The stronger this nonzero phase locking is, the larger PLI will be.
    Note that PLI does no longer indicate, which of the two signals is leading
    in phase. If PLI > 0, signal 1 is lagging; if PLI < 0, signal 1 is leading.

    """

    phase_diff = ((phase1 - phase2) + np.pi) % (2 * np.pi) - np.pi
    pli = np.mean(np.sign(phase_diff))
    h = np.histogram(phase1 - phase2, bins=100)
    pdiff = h[1][np.argmax(h[0])]
    
    return pli, pdiff

# =============================================================================
#                   Calculate coherence for an event
# =============================================================================

class GetEvent:
    
    def __init__(self, data, binsize, fs, iter_freqs, method):
        """

        Parameters
        ----------
        processed : dataframe
        binsize : float, seconds
        fs : int, sampling frequency (hz)
        iter_freqs : list, containing 2 element tuples with freq name and range
        method : str, type of coherence method

        Returns
        -------
        None.

        """
        # pass data to object
        self.processed = data
        self.fs = fs
        self.iter_freqs = iter_freqs
        self.binsize = int(binsize*fs)
        self.method = method
        
        # get grouped df by freq events and calculate time bins
        self.event = 'region'
        self.cols = ['band', ] #'low_freq', 'high_freq'
        self.processed = pd.DataFrame(self.processed.groupby(self.cols+[self.event])['phase'])
        self.bins = np.arange(0, self.processed.iloc[0,1].shape[0]-self.binsize, self.binsize)
        
        # get appropriate function
        if self.method == 'coh':
            self.func = spectral_coherence
        elif self.method == 'plv':
            self.func = phase_locking_value
        elif self.method =='pli':
            self.func = phase_lag_index
        

    def one_freq(self, signal1, signal2):
        """
        Calculate operation for each binsize.
    
        Parameters
        ----------
        signal1 : array,
        signal2 : array, 
    
        Returns
        -------
        vals : list,
    
        """
        vals = []
        for x in self.bins:
            val = self.func(signal1[x:x+self.binsize], signal2[x:x+self.binsize])
            vals.append(val)
        return vals
    
    
    def all_freqs(self):
        """
        Calculate for all freqs of an event.
        
        Parameters
        ----------
        df : dataframe, with filtered signal

        Returns
        -------
        out : dataframe, 

        """
        
        # prepare dataframe for getting dset
        df = self.processed
        df[self.cols+[self.event]] = df[0].apply(pd.Series)
        df = df.rename(columns={1:'phase'}).drop(labels=0, axis=1)
        
        # get coherence for each freq band
        bands = []
        for band in df.band.unique():
            dset = df.loc[df.band==band]
            vals = self.one_freq(dset.phase.iloc[0].values, dset.phase.iloc[1].values) 
            df_band = pd.DataFrame({'coherence':vals})
            df_band[self.cols] = dset[self.cols].iloc[0]
            bands.append(df_band)
            
        # create dataframe
        out = pd.concat(bands)
        out.insert(0, 'method', len(out)*['plv'])
        if self.method == 'pli':
            cols = ['coherence', 'phase_diff']
            out[cols] = out['coherence'].apply(pd.Series).rename(columns={0:cols[0], 1:cols[1]})
            out['method'] = 'pli'
            
        out.insert(1,'time', value=np.tile(self.bins/self.fs, len(out.band.unique())))
        
        return out


def bin_event_coh(df, binsize, fs, iter_freqs):
    """
    Calculate coherence for all freqs of an event.
    
    Parameters
    ----------
    df : dataframe, with filtered signal
    binsize: int, bin window in seconds
    fs : float, sampling frequency per second
    iter_freqs : list, containing 2 element tuples with freq name and range

    Returns
    -------
    out : dataframe, 

    """

    # get raw signals from dataframe
    signals = [x[1].iloc[0] for x in list(df.groupby('brain_region')['signal'])]
    
    # bin data based on user specifications  
    window_size = int(5*fs/2)
    noverlap = int(.5*window_size)
    binsize = int(binsize*fs)
    bins = np.arange(0, signals[0].shape[0]-binsize, binsize)
    
    # calculate coherence for each timebin (specified by user)
    vals = []
    for x in bins:
        val = spectral_coherence(signals[0][x:x+binsize], signals[1][x:x+binsize], fs, window_size, noverlap)
        vals.append(val)
        
    # create dataframe
    out = pd.DataFrame({'coherence': np.asarray([x[1] for x in vals]).flatten()})
    out.insert(0,'freq', value=np.asarray([x[0] for x in vals]).flatten())
    out.insert(0,'time', value=np.repeat(bins/fs, len(out.freq.unique())))
    
    out['method'] = 'coh'
    out[['band']] = '' #'low_freq', 'high_freq', 
    for band in iter_freqs:
        
        idx = out['freq'].between(band[1], band[2], inclusive='both')
        # out.loc[idx,'low_freq'] = band[1]
        # out.loc[idx,'high_freq'] = band[2]
        out.loc[idx,'band'] = band[0]

    out = out[out['band'] !='']
    out = out.groupby([ 'method','time','band', ])['coherence'].mean().reset_index()
    # out = out.groupby([ 'time', 'band', 'low_freq', 'high_freq',])['coherence'].mean().reset_index()
    return out


# =============================================================================
#                   Filter and calculate coherence for all events
# =============================================================================
def coherence_batch(downsampled_df, iter_freqs, fs, binsize, method='coh'):
    """
    Filter and calculate MI for all events.

    Parameters
    ----------
    downsampled_df : dataframe, contains signals
    iter_freqs : list, containing 2 element tuples with freq name and range
    fs : float, sampling frequency per second
    windowsize : float, seconds
    method: list, ['coh', 'plv', 'pli']

    Returns
    -------
    coh_df : dataframe, 
    """
    
    # get group columns
    group_cols = list(downsampled_df.columns[downsampled_df.columns.get_loc('stop_time')+1 \
                                              : downsampled_df.columns.get_loc('brain_region')])
        
    # convert to dframe for storing
    freq_dframe = pd.DataFrame.from_dict(iter_freqs)
    freq_dframe.columns = ['band_name', 'low_freq','high_freq',]
    
    # create grouped df
    animal_df = list(downsampled_df.groupby(['animal_id','file_name','start_time']))
    
    coh_df = []
    for (animal, file_name, start_time), data in tqdm(animal_df, desc='Calculating Coherence:'):

        # calculate spectral coherence
        coh_methods = []
        if 'coh' in method:
            coh = bin_event_coh(data, binsize, fs, iter_freqs)
            coh_methods.append(coh)
        # filter
        if ('plv' in method) | ('pli' in method):
            processed = BatchFilter(fs).filter_eventdf(data, freq_dframe)
            
        # calculate coherence
        if 'plv' in method:
            obj = GetEvent(processed, binsize, fs, iter_freqs, method='plv')
            coh = obj.all_freqs()
            coh_methods.append(coh)
            
        if 'pli' in method:
            obj = GetEvent(processed, binsize, fs, iter_freqs, method='pli')
            coh = obj.all_freqs()
            coh_methods.append(coh)
    
        # add group columns and append to list
        coh_methods = pd.concat(coh_methods)
        coh_methods.insert(0, 'animal', value=animal)
        # coh_methods.insert(1, 'start_time', value=start_time)
        coh_methods.reset_index(inplace=True, drop=True)
        coh_methods[group_cols] = data[group_cols].iloc[0]
        coh_df.append(coh_methods)

    data = pd.concat(coh_df)
    s = data.pop('coherence')
    data = pd.concat([data, s], axis=1)
        # pli phase_diff place last in df
    return  data.reset_index(drop=True)
        
        

if __name__ =='__main__':
    
    from preprocess import batch_downsample
    
    path = r'C:\Users\panton01\Desktop\test_coherence'
    new_fs = 250
    iter_freqs = [('lowtheta', 3, 6),('hightheta', 6, 12),\
                  ('beta', 13, 30),('gamma', 30, 80)]

    downsampled_df = batch_downsample(path, 'index_verified.csv', new_fs=new_fs)
    
    #specify 'standard' frequency/bands

    coh_df = coherence_batch(downsampled_df, 
                              iter_freqs, new_fs, 30, method= ['plv']) # ['coh', 'plv', 'pli']
    

















