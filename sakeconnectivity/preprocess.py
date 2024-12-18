# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
import multiprocessing
from joblib import delayed, Parallel
import adi
##### ------------------------------------------------------------------- #####

# =============================================================================
#                   Basic downsampling and filtering functions
# =============================================================================

def downsample(array, fs, new_fs):
    """
    Downsamples array.

    Parameters
    ----------
    array : array, 1D signal
    fs : int, sampling frequency
    new_fs : int, new sampling frequency

    Returns
    -------
    ds_array : array, 1D downsampled signal

    """

    sample_rate = int(fs)
    ds_factor = int(sample_rate/new_fs)
    ds_array = signal.decimate(array.astype(np.float32), ds_factor, ftype='fir')
    return ds_array


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Bandpass array using butterworth filter.

    Parameters
    ----------
    data : array, 1D signal
    lowcut : float, lower frequency cutoff
    highcut : float, upper frequency cutoff
    fs : int, sampling rate
    order : int, filter order. The default is 2

    Returns
    -------
    filt_data : array, 1D filtered signal

    """
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filt_data = signal.filtfilt(b, a, data)
    
    return filt_data


# =============================================================================
#                           Batch downsampling
# =============================================================================

def batch_downsample(main_path, idfile, new_fs=250):
    """
    Takes in the file name for a labchart and verified index (SAKEPlan output) csv. 
    Will read both files, downsample (based on user specifications), and construct a
    dataframe with downsampled signal. 

    Parameters
    ----------
    main_path : str, path to parent directpry
    idfile : pandas df, labchart and verified index.csv file names (strings)
    new_fs : int, new fs for downsampling

    Returns
    -------
    downsampleddf : pandas df, with downsampled signal

    """

    # read index file
    index_df = pd.read_csv(os.path.join(main_path, 'index_verified.csv'), sep=",")
    verified_cols = ['time_rejected', 'folder_path', 'animal_id']
    index_df[verified_cols] = index_df[verified_cols].fillna('')
    
    # create copy dataframe to store data
    downsampleddf = index_df
    downsampleddf['signal'] = 0
    downsampleddf['signal'] = downsampleddf['signal'].astype(object)
    for i, row in tqdm(index_df.iterrows(), desc='Downsampling:', total=len(index_df)):

        # get data and sampling rate
        fread = adi.read_file(os.path.join(main_path, row['folder_path'], row['file_name']))
        ch_obj = fread.channels[row['channel_id']]
        ch_data = ch_obj.get_data(row['block']+1, start_sample=int(row['start_time']), 
                                     stop_sample=int(row['stop_time']))
        
        # downsample the data and pass to dataframe
        downsampled = downsample(ch_data, row['sampling_rate'], new_fs)
        downsampleddf.at[i, 'signal'] = downsampled

    return downsampleddf

# =============================================================================
#                           Event filtering class
# =============================================================================

class BatchFilter():
    """
    Filter event df.
    """
    def __init__(self, fs):
        self.fs = fs

    def lfpextract(self, df):
        """
        Takes in an event dataframe and bandpasses the signal and calculates 
        instantnious phase and amplitutdes are calculated.
    
        Parameters
        ----------
        event_df : dataframe, containing the user-specified band start/stops
    
        Returns
        -------
        lfp_df : dataframe with the raw and filtered signal, as well as
                    the hilbert transform and amplitude values for filtered sig.
        """
    
        out = butter_bandpass_filter(data=df['signal'].iloc[0], 
                                     lowcut=df['low_freq'].iloc[0], 
                                     highcut = df['high_freq'].iloc[0],
                                     fs=self.fs, order=2)
        lfp_df = pd.DataFrame(out, columns=['filteredsig'])
        lfp_df.insert(0,'rawsig', value=df['signal'].iloc[0])
        lfp_df.insert(1,'region', value=df['region'].iloc[0])
        lfp_df.insert(2,'band', value=df['band_name'].iloc[0])
        lfp_df.insert(3,'low_freq', value=df['low_freq'].iloc[0])
        lfp_df.insert(4,'high_freq', value=df['high_freq'].iloc[0])
        hilbert_transform = signal.hilbert(lfp_df.filteredsig) 
        lfp_df['amp'] = np.abs(hilbert_transform)
        lfp_df['phase'] = np.angle(hilbert_transform, deg=False)

        return lfp_df

    def filter_eventdf(self, data, freq_dframe):
        """
        Filters event df.
    
        Parameters
        ----------
        data : dataframe, signal, frequency band dataframe
        freq_dframe : dataframe, frequencies to use for calculation
    
        Returns
        -------
        processed : dataframe, with filtered signal
        """
        
        # format freq dataframe with repeated signal for row
        region_set = []
        for i, row in data.iterrows():
            temp_df = freq_dframe.copy()
            temp_df['signal'] = [row['signal']] * len(freq_dframe) 
            temp_df['region'] = row['brain_region']
            region_set.append(temp_df)
        region_set = pd.concat(region_set)
        
        # group df and run parallel
        event_df = region_set.groupby(['region','band_name','low_freq','high_freq']) 
        func = self.lfpextract
        retLst = Parallel(n_jobs=int(multiprocessing.cpu_count()*.8))(delayed(func)(group) for name, group in event_df)

        return pd.concat(retLst)


if __name__ == '__main__':
    path = r'\\SUPERCOMPUTER2\Shared\acute_allo'
    new_fs=250
    #specify 'standard' frequency/bands
    iter_freqs = [('lowtheta', 3, 6),('hightheta', 6, 12),\
                  ('beta', 13, 30),('gamma', 30, 80)]

    #downsampled_df = loaddownsample(path, 'index_verified.csv')
    downsampled_df = pd.read_pickle('downdf.pickle')
        
    # covert to dframe for storing
    freq_dframe = pd.DataFrame.from_dict(iter_freqs)
    freq_dframe.columns = ['band_name','low_freq','high_freq']

