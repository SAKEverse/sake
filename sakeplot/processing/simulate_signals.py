####----------------------- IMPORTS ------------------- ######
import numpy as np
from processing.stft import get_freq_index, Stft, Properties
####--------------------------------------------------- ######

class SimSignal(Properties):
    """ Simulate eeg/lgp signals
    """
    
    def __init__(self, properties:dict, time_duration:np.ndarray):
        
        # pass parameters to object
        super().__init__(properties)
        self.time_duration = time_duration
        
        # create time vector
        self.t = np.arange(0, self.time_duration, 1/self.sampling_rate)
    
    def make_sine(self, freq:float, amp:float):
        return np.sin(freq*self.t*np.pi*2) * amp
        
    def make_sine_norm(self, freq:float, amp:float, rhythm):
        """
        Create normally distributed sine wave

        Parameters
        ----------
        amp : float, amplitude
        freq : float, frequency
        rhythm : float, rhythmicity 0 to inf

        Returns
        -------
        signal : np.ndarray, 1d signal

        """

        # create template wave
        template = np.sin(freq*self.t[0:int(np.ceil(self.sampling_rate/freq))+1]*np.pi*2)

        # create normaly distributed events
        mu = 1/freq
        sigma = mu/(rhythm + (1^-10))
        n_events = int(np.ceil(self.t.shape[0]/template.shape[0]))
        s = np.random.normal(mu, sigma, int(n_events *1.2))

        # get inter event interval and find index
        isi = np.cumsum(s)
        index = get_freq_index(self.t, isi)

        # create logic vector to be convolved
        logic_vector = np.zeros(self.t.shape)
        logic_vector[index] = 1

        # return convolved signal
        return np.convolve(logic_vector, template, mode = 'same') * amp

    def add_sines(self, freq:list, amp:list):
        """
        Add multiple sine waves

        Parameters
        ----------
        freq : list
        amp : list

        Returns
        -------
        signal :  np.ndarray, 1d signal

        """

        signal = np.zeros(self.t.shape)
        for f,a in zip(freq, amp):
           signal += self.make_sine(f, a)
        return signal
    
    def add_sines_norm(self, freq:list, amp:list, rhythm:list):
        """
        Add multiple sine waves

        Parameters
        ----------
        freq : list
        amp : list

        Returns
        -------
        signal :  np.ndarray, 1d signal

        """

        signal = np.zeros(self.t.shape)
        for i in range(len(freq)):
           signal += self.make_sine_norm(freq[i], amp[i], rhythm[i])
        return signal


if __name__ == '__main__':
    from matplotlib.pyplot import plot
    properties = {'sampling_rate':4000, 'fft_win':5, 'freq_range': [5, 121], 
                'fft_overlap':0.5, 'mains_noise': [59, 61]}
    
    freq = [10, 60]
    amp = [5, 10]
    rhythm = [15, 15]
    time_duration = 30              # in seconds
    obj = SimSignal(properties, time_duration)
    signal = obj.add_sines_norm(freq, amp, rhythm) + obj.add_sines([60], [3])
    # plot(signal)

    # get power
    stft_obj = Stft(properties)
    freq_vector, pmat = stft_obj.get_stft(signal)
    
    # plot original
    plot(freq_vector, np.mean(pmat, axis = 1))
    
    # remove noise
    pmat = stft_obj.remove_mains(freq_vector, pmat)
    
    # plot noise removed PSD
    plot(freq_vector, np.mean(pmat, axis = 1))














