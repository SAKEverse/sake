# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from joblib import delayed, Parallel
import multiprocessing
from preprocess import BatchFilter
# import matplotlib.pyplot as plt
# import seaborn as sns
##### ------------------------------------------------------------------- #####

# =============================================================================
#           Functions and class to calculate Phase Amp Index
# =============================================================================


def tort_mi(phase_array, amp_array):
    """
    Calculate Tort MI (Tort, 2010).

    Parameters
    ----------
    phase_array : array, instantenous phase
    amp_array : array, instantenous amp

    Returns
    -------
    ampmeans : array, amplitude for each bin
    phasebins : array phase for each bin
    MI : array, array, MI
    phase_array : array, instantenous phase
    amp_array : array, instantenous amp
    """
    
    # define phase bins
    phasebins = np.arange(-np.pi, np.pi, 0.3)
    L = phasebins.shape[0] - 1  
    
    # # create phase amps
    # phase_amps = pd.Series(data=amp_array, index=phase_array)
    
    # # get bin range for each phase
    # idx = pd.Series(pd.cut(phase_amps.index, phasebins))
    
    # # groupby bin range
    # ampmeans = phase_amps.groupby(idx).mean()
    ampmeans = np.zeros(L)
    idx = np.digitize(phase_array, phasebins, right=False)-1
    # print(idx.unique())
    for k in range(L):
        ampmeans[k] = np.mean(amp_array[idx==k])
    
    # Modulation Index
    amplP = ampmeans/np.sum(ampmeans)
    amplQ = np.ones(L)/(L)
    distKL = np.sum(amplP*np.log(amplP/amplQ))
    MI = distKL/np.log(L)
      
    # L = phasebins.shape[0] - 1                                  # get number of phase bins
    # ampmeans = np.zeros(L)                                      # vector to hold amplitude
    # # get index of phase bins
    #---------------------------------
    # idx = np.digitize(phase_array, phasebins, right=False)-1
    #---------------------------------
    # # get mean for each bin
    # for k in range(L):
    #     ampmeans[k] = np.mean(amp_array[idx==k])

    return ampmeans, phasebins, MI, phase_array, amp_array

class PhaseAmp:
    """
    Calculate phase amplitude coupling for an event.
    """
    
    def __init__(self, event_df, windowsize=30, fs=250):

        self.windowsize = windowsize
        self.fs = fs
        self.event_df = event_df.groupby(['bandcomp', 'regioncomp'])
        self.njobs = int(multiprocessing.cpu_count()*.8)
        
    def run_parallel(self):
        """
        Parameters
        ----------
        Returns
        -------
        dataframe of concatenated dataframes which have had function applied
        """
        retLst=[]
        for name, group in self.event_df:
            retLst.append(self.get_event(group))
        # retLst = Parallel(n_jobs=self.njobs)(delayed(self.get_event)(group) for name, group in self.event_df)
        return pd.concat(retLst)
    
    def get_event(self, df):
        """
        Calculate MI for all freqs of an event.
        
        Parameters
        ----------
        df : dataframe, with filtered signal

        Returns
        -------
        out : dataframe, 

        """
        
        # convert into long format
        signals = [np.asarray(df.phase.iloc[0]), np.asarray(df.amp.iloc[0])]
        
        # bin data based on user specifications   
        window_size = int(self.windowsize*self.fs)
        bins = np.arange(0, signals[0].shape[0]-window_size, window_size)
    
        vals = [tort_mi(signals[0][x:x+window_size], signals[1][x:x+window_size])\
                    for x in bins]
            
        # create dataframe
        out = pd.DataFrame([x[2] for x in vals], columns=['MI'])
        out.insert(0,'band', value=df.bandcomp.unique()[0])
        out.insert(0,'direction', value=df.regioncomp.unique()[0])
        out.insert(0,'time', value=[bins[idx]/int(self.fs) for idx in range(len(bins))])
        # out.insert(1,'amp_inphase', value=[x[0] for x in vals])
        # out.insert(2,'phase_bins', value=[x[1][:-1] for x in vals])
        # out.insert(6,'lowfreq_phase', value = [x[3] for x in vals])
        # out.insert(7,'highfreq_amp', value = [x[4] for x in vals])
        
        return out


# =============================================================================
#           Functions and class to calculate Phase Amp Index
# =============================================================================

def get_filter_df(iter_freqs, event_df):
    """
    Takes in a dataframe containing the frequency bands to analyze, along
    with the filtered signal dataframe (containing the phase/amplitude values)
    and creates a datagrame in tidy-form used for a groupby function.

    Parameters
    ----------
    iter_freqs : List, contains tuples with frequency name and band
    event_df : dataframe, containing signal

    Returns
    -------
    dataframe, with filtered signal for each frequency band and direction 

    """
  
    # create list of all combinations of frequency bands to compare
    bandlist = list(itertools.combinations(iter_freqs, 2))
    
    # ensure that combination-tubples always have low-freq before high-freq
    bandlist = [x if x[0][1]<x[1][1] else (x[1],x[0]) for x in bandlist]
    
    out = []
    for bandset in bandlist:
        for regionset in zip([event_df.region.unique(),event_df.region.unique()[::-1]]):
            temp_df =  pd.DataFrame(['%s_%s' %(bandset[0][0],bandset[1][0])],columns=['bandcomp'])
            temp_df.insert(1,'regioncomp',value = '%s_%s' %(regionset[0][0],regionset[0][1])) 
            temp_df.insert(2,'phase',value = [np.asarray(event_df.loc[(event_df.region==regionset[0][0])\
                                    &(event_df.band==bandset[0][0])]['phase'])]) 
            temp_df.insert(3,'amp',value = [np.asarray(event_df.loc[(event_df.region==regionset[0][1])\
                                    &(event_df.band==bandset[1][0])]['amp'])]) 
            temp_df.insert(4,'filt',value = [np.asarray(event_df.loc[(event_df.region==regionset[0][1])\
                                        &(event_df.band==bandset[1][0])]['filteredsig'])]) 
            out.append(temp_df)

    return pd.concat(out)


# =============================================================================
#                   Filter and calculate MI for all events
# =============================================================================

def phaseamp_batch(downsampled_df, iter_freqs, fs, windowsize):
    """
    Filter and calculate MI for all events.

    Parameters
    ----------
    downsampled_df : dataframe, contains signals
    iter_freqs : list, containing 2 element tuples with freq name and range
    fs : float, sampling frequency per second
    windowsize : float, seconds

    Returns
    -------
    all_pa_couples : dataframe, 
    """
    
    # get group columns
    group_cols = list(downsampled_df.columns[downsampled_df.columns.get_loc('stop_time')+1 \
                                              : downsampled_df.columns.get_loc('brain_region')])

    # convert to dframe for storing
    freq_dframe = pd.DataFrame.from_dict(iter_freqs)
    freq_dframe.columns = ['band_name','low_freq','high_freq']
    
    # create grouped df
    animal_df = list(downsampled_df.groupby(['animal_id','file_name','start_time']))
    
    all_pa = []
    for (animal, file_name, start_time), data in tqdm(animal_df, desc='Calculating Phase Amp:'):
        
        # check that 2 rows of data were grouped
        if len(data) != 2:
            continue
        
        # filter
        processed = BatchFilter(fs).filter_eventdf(data, freq_dframe)
        phaseamp_df = get_filter_df(iter_freqs, processed)

        # calculcate MI
        pa_couples = PhaseAmp(phaseamp_df, windowsize=windowsize,
                              fs=fs).run_parallel()
        
        # add group columns
        pa_couples.insert(0, 'animal', value=animal)
        # pa_couples.insert(1, 'start_time', value=start_time)
        pa_couples.reset_index(inplace=True, drop=True)
        pa_couples[group_cols] = data[group_cols].iloc[0]
        all_pa.append(pa_couples)

    data = pd.concat(all_pa)
    s = data.pop('MI')
    data = pd.concat([data, s], axis=1)
    data.insert(1, 'method', len(data)*['tort'])
    return data.reset_index(drop=True)

if __name__ =='__main__':
    
    from preprocess import batch_downsample
    
    path =  r'C:\Users\panton01\Desktop\test_pac' # r'C:\Users\panton01\Desktop\test_coherence' r'\\SUPERCOMPUTER2\Shared\acute_allo' 
    new_fs = 250
    iter_freqs = [('lowtheta', 3, 6),('hightheta', 6, 12),\
                  ('beta', 13, 30),('gamma', 30, 80)]

    downsampled_df = batch_downsample(path, 'index_verified.csv', new_fs=new_fs)
    data = phaseamp_batch(downsampled_df, iter_freqs, new_fs, 30)
