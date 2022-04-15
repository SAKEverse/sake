# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pyedflib
##### ------------------------------------------------------------------- #####

def make_sine(t, freq:float, amp:float):
    """
    Create sine wave.

    Parameters
    ----------
    t : array, time
    freq : float, frequency
    amp : float, amplitude

    Returns
    -------
    array, sine wave

    """
    return np.sin(freq*t*np.pi*2) * amp

class CohWaves:
    def __init__(self, mod_array, win=5, 
                 fs=1000, freq=20,
                 amp=1, noise=.5, phase_shift=0):

        self.mod_array = mod_array
        self.win = win
        self.fs = fs
        self.freq = freq
        self.amp = amp
        self.noise = noise
        self.phase_shift = phase_shift
        
    def create_coh_waves(self, jitter, length=30):
        
        jitter 
        
        # phase shift up to 360
        phase_shift = self.phase_shift%360
    
        # get interever inteval in samples
        iei = int(self.fs/self.freq)
        
        # get phase and amplitude
        phase = np.linspace(-np.pi, np.pi, iei)
        amp = np.sin(phase)
        
        # create template and shifted waves
        orig_length = length
        trim = 1 *self.fs
        length = orig_length*self.fs + 2*trim
        template = np.zeros(length)
        shifted = np.zeros(length)
        
        # get wave index
        idx_temp = np.arange(0, template.shape[0]-trim, iei).astype(int)
    
        # get phase shift
        phase_shift_wave = np.linspace(0, 360, iei)
        const_phase_shift = np.argmin(np.abs(phase_shift_wave-phase_shift))
        m = iei*jitter  # mean (max should be cycle length)
        sd = m/5        # standard deviation
        var_shift =  np.random.normal(m, sd, len(idx_temp)).astype(int)
        idx_shifted = idx_temp + var_shift +const_phase_shift
        
        # convolve waves
        template[idx_temp] = 0*np.random.rand(len(idx_temp)) + np.ones(len(idx_temp))
        shifted[idx_shifted] = 0*np.random.rand(len(idx_temp)) + np.ones(len(idx_temp))
        noise_arrayt = np.random.rand(len(template)) * self.noise
        noise_arrays = np.random.rand(len(template)) * self.noise
        template = np.convolve(template, amp, mode='same') + noise_arrayt
        shifted = np.convolve(shifted, amp, mode='same') + noise_arrays
        
        return template[trim:-trim]*self.amp, shifted[trim:-trim]*self.amp
    
    
    def create_one_ch_set(self):
        """
        Create two channel data (pair of data for coherence)

        Returns
        -------
        tuple, (array, array) with signa

        """
    
        wave1 = []
        wave2 = []
        for jitter in self.mod_array:
           orig, shifted = self.create_coh_waves(jitter, length=self.win)
           wave1.append(orig)
           wave2.append(shifted)
        return np.concatenate(wave1), np.concatenate(wave2)


class ModWaves:
    """
    Create test waves for phase amplitude coupling.
    """
    
    def __init__(self, mod_array, win=5, 
                 fs=1000, freq=[8, 50],
                 amp=[1,1]):

        self.mod_array = mod_array
        self.win = win
        self.fs = fs
        self.freq = freq
        self.amp = amp
        
    def create_mod_wave(self, mod_level, length=10):
        """
        Create modulated waves.
     
        Parameters
        ----------
        mod_level : float, 0-1
        length : int, The default is 10.
        fs : int, The default is 1000.
        freq : list, The default is [8, 50].
        amp : list,  The default is [1, 1].
     
        Returns
        -------
        x : array, phase of slow wave
        y : array, amp of fast wave
     
        """  
        if mod_level>1:
            mod_level=1
        mod_level = 1 - mod_level
        
        # get time vector
        dt = 1/self.fs
        t = np.arange(0,length, dt)
        
        # create phase wave
        x = self.amp[0]*np.sin(2*np.pi*t*self.freq[0])
        
        # create modulated amplitude
        Afa = 1*(((1-mod_level)*x)+mod_level+1)/2
        y = Afa*np.sin(2*np.pi*t*self.freq[1])*self.amp[1]
        
        # add noise
        x = x + np.random.rand(x.shape[0]) *1
        return x, y

    def create_one_ch(self):
        """
        Create one channel data

        Returns
        -------
        1d array, wave

        """
    
        wave = []
        for mod in self.mod_array:
           x,y = self.create_mod_wave(mod, length=self.win)
           wave.append(x+y)
        return np.concatenate(wave)
           


class EdfMaker:
    """
    Create EDFs to test seizure detection.
    """
    
    def __init__(self, properties):
        
        # get values from dictionary
        for key, value in properties.items():
               setattr(self, key, value)
        
        # get channel numbers in python format
        self.animal_ch_list = np.array(self.animal_ch_list) - 1
        
        # create time vector
        self.t = np.arange(0, len(self.mod_arrays[0])*self.win, 1/self.fs)
        
        # get info for channels
        self.channel_info = []

        for ch_list in self.animal_ch_list:
            for i, ch in enumerate(ch_list):
                # get channel properties
                ch_dict = {'label': self.ch_id[i], 'dimension': self.dimension[i], 
                   'sample_rate': self.fs, 'physical_max': self.physical_max[i],
                   'physical_min': self.physical_min[i], 'digital_max': self.digital_max[i], 
                   'digital_min': self.digital_min[i], 'transducer': '', 'prefilter':''}
                
                # append info
                self.channel_info.append(ch_dict)
           
    def create_data_mod(self):
        """
        Create data for all animals in a file.

        Returns
        -------
        data: list, containing arrays for each edf channel.

        """
        
        data = []
        
        # iterate over animal lists
        for ch_list, mod_array in zip(self.animal_ch_list, self.mod_arrays):
            
            # iterate over channels for each animal
            for ch in ch_list:
                mod_obj = ModWaves(mod_array, win=self.win, 
                             fs=self.fs, freq=self.freq,
                             amp=self.amp)
                # append data for each channel
                data_ch = mod_obj.create_one_ch()
                data.append(data_ch * self.scale)  
                
        return data
    
    def create_data_coh(self):
        """
        Create data for all animals in a file.

        Returns
        -------
        data: list, containing arrays for each edf channel.

        """
        
        data = []
        
        # iterate over animal lists
        for ch_list, mod_array in zip(self.animal_ch_list, self.mod_arrays):
            
            # create coherence waves
            coh_obj = CohWaves(mod_array, win=self.win, fs=self.fs,
                               freq=self.freq, amp=self.amp, phase_shift=self.phase_shift)
            [ch1, ch2] = coh_obj.create_one_ch_set()
            
            data_chs = [ch1, ch2] + [np.random.rand(ch1.shape[0]) * (len(ch_list)-2)]
            # append data for each channel
            for ch in data_chs:
                data.append(ch * self.scale)  
                
        return data
    
    def create_edf(self,):
        """
        Create_edf file.
        
        Parameters
        ----------

        Returns
        -------
        None.
        """
        

        # intialize EDF object
        file_name = 'sim_' + '_'.join(self.animal_ids)
        file_path = os.path.join(self.save_path, file_name+ '.edf')
        
        with pyedflib.EdfWriter(file_path, self.animal_ch_list.size,
                                file_type = pyedflib.FILETYPE_EDF) as edf:
            
            # write headers
            edf.setSignalHeaders(self.channel_info)
                
            # write data to edf file based on method
            if self.simtype == 'pac_modulation':
                data = self.create_data_mod()
            elif self.simtype == 'coherence':
                data = self.create_data_coh()
            edf.writeSamples(data)
    
            # close edf file writer
            edf.close()
        print('Edf was created:', file_path)
    
    

if __name__ == '__main__':
    
    mod_array_len = 10 # seconds
    properties = {"save_path": r"C:\Users\panton01\Desktop\test_coherence",
        "fs" : 1000,
        "ch_id":  ["lfp","eeg","emg"],
        "dimension": ["V","V","V"],
        "physical_max": [0.1, 0.1, 0.01],
        "physical_min": [-0.1, -0.1, -0.01],
        "digital_max": [32000, 32000, 32000],
        "digital_min": [-32000, -32000, -32000],
        "scale": 1e-3,
        "animal_ch_list":  [[1,2,3], [4,5,6], [7,8,9], [10,11,12]],
        "animal_ids":  ['mouse1', 'mouse2', 'mouse2', 'mouse4'],
        "win": 30,
        }

    # # PAC
    # properties.update({ "simtype": "pac_modulation",      
    #                    "freq": [8, 50], "amp": [1, 1],
    #                     "mod_arrays": [np.linspace(.1,.9, mod_array_len), 
    #                                     np.linspace(.4,.8, mod_array_len),
    #                                     np.linspace(.9,.1, mod_array_len),
    #                                     np.linspace(.4,.4, mod_array_len),],
    #                     })    

    # Coherence
    properties.update({"simtype": "coherence", 
                    "freq": 20, "amp": 1, "phase_shift": 90,
                    "mod_arrays": [np.linspace(2, 0, mod_array_len), 
                                        np.linspace(0, 2, mod_array_len),
                                        np.linspace(1.5,1.5, mod_array_len),
                                        np.linspace(.2,.2, mod_array_len),],
                        })
    
    # Create EDF
    # obj = EdfMaker(properties)
    # obj.create_edf()
    
    # =============================================================================
    #               Plot pac
    # =============================================================================
    # import matplotlib.pyplot as plt
    # a = np.arange(0, 1.1, .1)
    # obj = ModWaves(np.array([]))
    # f,axs = plt.subplots(nrows=len(a))
    # axs = axs.flatten()
    # for i,mod in enumerate(a):
    #     x,y = obj.create_mod_wave(mod, length=1)
    #     axs[i].plot(y, label=str(mod))
    #     axs[i].legend()
        
    # =============================================================================
    #               Plot coherence
    # =============================================================================
    # import matplotlib.pyplot as plt
    # a = np.arange(0, 1, .3)
    # obj = CohWaves(np.array([]))
    # f,axs = plt.subplots(nrows=len(a))
    # axs = axs.flatten()
    # for i,coh in enumerate(a):
    #     x,y = obj.create_coh_waves(coh, length=1)
    #     axs[i].plot(x)
    #     axs[i].plot(y, label=str(coh))
    #     axs[i].legend()
    # # =============================================================================
    #               # Plot simulated data
    # =============================================================================
    # obj = EdfMaker(properties)
    # data = obj.create_data_coh()
    
    # import matplotlib.pyplot as plt
    # from matplotlib.pyplot import cm
    
    # color = cm.Paired(np.linspace(0, 1, len(data)))
    # f,axs = plt.subplots(len(data),1, sharex=True)
    
    # for i, (ax, data_ch) in enumerate(zip(axs, data)):
    #     ax.plot(obj.t, data_ch, c=color[i,:])
            
            
            
    

        
            
 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            