# -*- coding: utf-8 -*-
########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import contextlib
import joblib
from tqdm import tqdm
import psutil
from load.get_data import AdiGet
from processing.stft import Stft, Properties
########## ---------------------------------------------------------------- ##########

def rem_array(start, stop, div):
    """
    Make array with remaining as last item.
    
    Parameters
    ----------
    start : int
    stop : int
    div : int

    Returns
    -------
    idx_array : numpy array

    """

    rem = stop % div
    if rem == stop:
        idx_array = np.array([start,stop])
    else:
        trim_stop = (stop - rem)
        idx_array = np.arange(start, trim_stop, div, dtype = int)
        idx_array = np.append(idx_array, stop)
    return idx_array

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress
    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

class BatchStft():
    """
    Batch analysis of stft signals using index from SAKE.
    """
    
    def __init__(self, properties, index_df, njobs=None):
        
        self.properties = properties
        self.index_df = index_df

        self.max_cpu_load = 0.8
        self.max_mem_load = 0.1
        
        # get cpu count
        max_jobs = int(np.floor(multiprocessing.cpu_count() * self.max_cpu_load))

        if njobs:
            self.njobs = njobs
        else:
            self.njobs = max_jobs
            
        if self.njobs > max_jobs:
            self.njobs = max_jobs
        
        # get chunksize based on available memory and cpu count
        byte_per_sample = 5
        mem = int(psutil.virtual_memory().available * self.max_mem_load)
        self.chunksize = int(mem/max_jobs/byte_per_sample)
        
        # drop rows containing NaNs after filling folder_path and animal_id
        verified_cols = ['time_rejected', 'accepted']
        columns = ['folder_path', 'animal_id']

        if set(verified_cols).issubset(self.index_df.columns):
            columns += verified_cols
        self.index_df[columns] = self.index_df[columns].fillna('')
        self.index_df = self.index_df.dropna().reset_index(drop=True)
        
    def get_pmat_batch(self):
        """
        Get power for each row of index_df.

        Returns
        -------
        power_df : pandas dataframe, with each row containing freq and power

        """
        
        power_df = pd.DataFrame(np.empty((len(self.index_df), 2)), columns = ['freq', 'pmat'], dtype = object)
        
        print('\n--> Processing with', self.njobs, 'thread(s):\n')

        if self.njobs == 1:
            lst = []
            for idx, row in self.index_df.iterrows():
                lst.append(self.get_pmat(idx, row))
        else:
            with tqdm_joblib(tqdm(desc='Extracting Power', total=len(self.index_df))) as progress_bar:  # NOQA
                lst = Parallel(n_jobs=self.njobs, backend='loky')(delayed(self.get_pmat)(idx, row) for idx, row in self.index_df.iterrows())
            
        for i, freq, pmat in lst:
            power_df.at[i, 'freq'] = freq
            power_df.at[i, 'pmat'] = pmat
        return power_df
    
    def get_pmat(self, i, row):
        """
        Get power from one index df row.

        Parameters
        ----------
        i : int, loc in index df
        row : series, row of index df

        Returns
        -------
        i : int, loc in index df
        freq : array, freq range
        pmat : array, power(freq, time)

        """
        
        # get properties
        file_properties = self.index_df[AdiGet.input_parameters].loc[i].to_dict()
        file_properties.update({'search_path': self.properties['search_path']})
        
        # add sampling rate
        self.properties.update({'sampling_rate':int(file_properties['sampling_rate'])})
        
        # Init Stft object with required properties
        selected_keys = Properties.types.keys()
        selected_keys = list(selected_keys)
        
        # select key-value pairs from dictionary
        selected_properties = {x: self.properties[x] for x in selected_keys}
        
        # convert time series to frequency domain
        stft_obj = Stft(selected_properties)
        freq = stft_obj.f
        
        # create index
        chunksize = self.chunksize//stft_obj.sampling_rate*stft_obj.sampling_rate
        temp_idx = rem_array(file_properties['start_time'], 
                   file_properties['stop_time'],
                   chunksize)
        idx = np.zeros((len(temp_idx)-1, 2), dtype=int)
        idx[:, 0] = temp_idx[:-1]
        idx[:, 1] = temp_idx[1:]
        idx[:, 0] -= stft_obj.fft_overlap_size
        idx[:-1, 1] -= 1
        
        # get power for each segment
        pmat = np.zeros((freq.shape[0], 0))
        for ii in range(idx.shape[0]):
            signal = AdiGet(file_properties).get_data_adi(start=idx[ii,0], stop=idx[ii,1])
            pmat_temp = stft_obj.run_stft(signal)[:,1:-1]
            pmat = np.concatenate((pmat, pmat_temp),axis=1)
        pmat = np.concatenate((pmat, pmat_temp[:,-1:]),axis=1)

        # print('--> Analyzed File - ' + str(i+1) + '. Total:' + str(len(self.index_df)) +  '.\n')
        return i, freq, pmat
    
    
if __name__ == '__main__':
    import os, yaml
    from load_index import load_index
    
    def load_yaml(settings_path):
        with open(settings_path, 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    
    ### ---------------------- USER INPUT -------------------------------- ###
    
    ## define path and conditions for filtering
    parent_folder = r'\\SUPERCOMPUTER2\Shared\acute_allo'
    settings_yaml = 'settings.yaml'
    load_path_yaml = 'path.yaml'
    path = load_yaml(load_path_yaml)
    properties = load_yaml(settings_yaml)
    properties.update({'search_path': path['search_path']})
    ### ---------------------------------------------------------------- ####
    
    ## load data frame
    index_df = load_index(os.path.join(parent_folder, 'index.csv'))
    index_df = index_df.fillna('')
    index_df = index_df.dropna().reset_index(drop=True)
   
    obj = BatchStft(properties, index_df, 1) 
    power_df = obj.get_pmat()
    # i=0
    # obj.get_pmat(i, index_df.loc[0])
    

    
    # import matplotlib.pyplot as plt
    # def make_sine(t, freq, amp=1):
    #     return np.sin(freq*t*np.pi*2) * amp
        
    # fs=4000
    # time_dur = 999
    # step= stft_obj.fft_win*stft_obj.fft_overlap
    # t = np.arange(0, time_dur, 1/fs)
    # freqs = np.arange(5,30,5, dtype=int)
    
    # wave = np.array([])
    # for f in freqs:
    #     wave = np.concatenate((wave,  make_sine(t, f, amp=1)))      
    # pmat_one = stft_obj.run_stft(wave)
    
    # pmat_conc = np.zeros((pmat_one.shape[0],0))
    # for f in freqs:
    #     wave = make_sine(t, f, amp=1)
    #     pmat_conc = np.concatenate((pmat_conc,  stft_obj.run_stft(wave)), axis=1)  
    
    # (f,axs) = plt.subplots(2,1,)
    # tpmat = np.arange(0, pmat_one.shape[1]*step, step)
    # axs[0].plot(tpmat, 1+np.argmax(pmat_one,axis=0)/stft_obj.fft_win, '-x')
    # tpmat = np.arange(0, pmat_conc.shape[1]*step, step)
    # axs[1].plot(tpmat, 1+np.argmax(pmat_conc,axis=0)/stft_obj.fft_win, '-x')
    
    
    
    
    
    
    
    
    
    
    
    
    
