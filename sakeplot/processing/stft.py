########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
import pandas as pd
from typing import Union#, List
from beartype import beartype
from scipy.signal import stft as scipy_stft
########## ---------------------------------------------------------------- ##########


class GetIndex():
    "Get index"
    
    def __init__(self, array):
        self.array = array

    def find_nearest(self, value):
        """
        Find nearest value in self.array.

        Parameters
        ----------
        value : values to search the array for

        Returns
        -------
        idx for values
        
        """
        return (np.abs(self.array - value)).argmin()

@beartype
def get_freq_index(freq_vector:np.ndarray, freqs) -> np.ndarray: #freqs: List[Union[int, float]]
    """
    Get frequency index.

    Parameters
    ----------
    freq_vector : np.ndarray, frequency vector to be indexed
    freqs : values to find the index

    Returns
    -------
    np.ndarray

    """
    
    # instantiate
    f = GetIndex(freq_vector)
    
    # vectorize function
    vfunc = np.vectorize(f.find_nearest)

    # return index 
    return vfunc(freqs)

def f_fill(arr:np.ndarray, axis:int = 0) -> np.ndarray:
    """
    Replace nans using pandas ffil method.

    Parameters
    ----------
    arr : np.ndarray
    axis : int, axis for filling operation

    Returns
    -------
    np.ndarray

    """
    # convert to dataframe and fill missing
    df = pd.DataFrame(arr)
    df = df.interpolate(method='nearest', limit_direction='forward', axis = axis)

    return  df.values


class Properties:
    " Convert dictionary to class properties and check types"
    
    types = {'sampling_rate':int, 'fft_win':int, 'fft_freq_range':list, 
             'fft_overlap':float, 'mains_noise':list}
    
    def __init__(self, properties:dict):
        # Add dictionary elements to object attributes if variable names and types match
        for key, value in properties.items():
            if key in self.types:
                if self.types[key] == type(value):
                    setattr(self, key, value)
                else:
                    raise(Exception('-> Got ' + str(type(value)) + '. Expected: ' + str(self.types[key])   + '.\n'))
            else:
                raise(Exception('-> Variable *' + key + '* was not found.\n'))

@beartype                          
def check_range_input(input_range:list, lower_limit : Union[float, int], upper_limit: Union[float, int]):
    """
    Check whether input_range list is valid

    Parameters
    ----------
    input_range: list
    lower_limit : (float|int).
    upper_limit : (float|int)

    Returns
    -------
    None.

    """
    
    # check that there are only two real numbers [lower, upper] limit within nyquist limit
    if all(isinstance(x, (int, float)) for x in input_range) == False:
        raise(Exception('-> Elements in list have to be numeric.\n'))
    if len(input_range) != 2:
        raise(Exception('-> Got length of freq_range : ' + str(len(input_range)) + '. Expected : 2.\n'))
    if input_range[0] > input_range[1]:
        raise(Exception('-> The second element of freq_range has to be greater than the first.\n'))
    if any(np.array(input_range) < lower_limit):
        raise(Exception('-> Values can not be below lower limit: ' + str(lower_limit) +'.\n'))
    if any(np.array(input_range) > upper_limit):
        raise(Exception('-> Values can not exceed upper limit: ' + str(upper_limit) +'.\n'))

                    
# Single PSD class
class Stft(Properties):
    """  
    Perform Stft analysis on 1D signal.

    """
    
    @beartype
    def __init__(self, properties:dict):
        """

        Parameters
        ----------
        sampling_rate : int
        fft_win : int
        fft_freq_range : list
        fft_overlap : float, the default is 0.5.

        Returns
        -------
        None.

        """

        # pass parameters to object
        super().__init__(properties)
        self.winsize = int(self.sampling_rate * self.fft_win)                   # window size (samples)  
        self.fft_overlap_size = int(self.winsize * self.fft_overlap)            # fft_overlap size (samples)
        self.f_idx = self.get_freq_idx(self.fft_freq_range)                     # get frequency index
        
        # check that there are only two real numbers [lower, upper] limit within nyquist limit
        check_range_input(self.fft_freq_range, 0, self.sampling_rate/2)
        
        # check if mains noise is within user specified frequency range
        check_range_input(self.mains_noise, self.fft_freq_range[0], self.fft_freq_range[1])
        
        # get frequency range
        self.f = np.arange(self.fft_freq_range[0], self.fft_freq_range[1] + (1/self.fft_win),  1/self.fft_win)
    
    @beartype
    def get_freq_idx(self, f:list) -> np.ndarray:
        """
        Convert frequency value to frequency index based on sampling rate

        Parameters
        ----------
        f : list, containing frequency value

        Returns
        -------
        freq_idx : list, frequency index value(int)

        """

        freq_idx = np.zeros(len(f), dtype = np.int32)
        for i in range(len(f)):
            freq_idx[i] = int(f[i]*(self.winsize/self.sampling_rate))
        
        return freq_idx
        
    @beartype
    def get_stft(self, input_wave:np.ndarray):
        """
        Run short time fourier transfrom on input_wave.

        Parameters
        ----------
        input_wave : np.ndarray, 1D signal

        Returns
        -------
        power_matrix : 2D numpy array, rows = freq and columns = time bins

        """
        
        # get spectrogram # f, t, pmat =
        _, _, pmat = scipy_stft(input_wave, self.sampling_rate, 
                                nperseg=self.winsize, 
                                noverlap=self.fft_overlap_size,
                                padded=False)

        # get real power
        pmat = np.square(np.abs(pmat[self.f_idx[0] : self.f_idx[1]+1,:]))
        
        return pmat #f[self.f_idx[0] : self.f_idx[1]+1], 

    @beartype
    def remove_mains(self, freq:np.ndarray, pmat:np.ndarray) -> np.ndarray:
        """
        Remove mains noise, using nans replacement and

        Parameters
        ----------
        freq : np.ndarray
        pmat : np.ndarray

        Returns
        -------
        pmat : np.ndarray

        """
        
        # find frequency index
        f_idx = get_freq_index(freq, self.mains_noise)

        # set noise index to NaNs
        pmat[f_idx[0]:f_idx[1]+1,:] = np.nan

        # fill NaNs
        pmat = f_fill(pmat, axis=0)

        return pmat
    
    @beartype
    def run_stft(self, input_wave:np.ndarray):
        """
        Get stft and remove mains noise.

        Parameters
        ----------
        input_wave : np.ndarray, 1D signal
        f_noise : list, lower and upper bounds of mains noise

        Returns
        -------
        freq : np.ndarray, real frequency vector
        pmat : np.ndarray, transformed spectogram

        """

        # get stft 
        pmat = self.get_stft(input_wave)

        # remove mains nose
        pmat = self.remove_mains(self.f, pmat)
        
        return pmat
        
        
        
        
        
        
        
        
        
        
        
        
        