####----------------------- IMPORTS ------------------- ######
import pytest
import numpy as np
from processing.simulate_signals import SimSignal
from processing.stft import get_freq_index, Stft
####--------------------------------------------------- ######
time_duration = 30 # for signal duration (in seconds)
####----------------------- Fixtures ------------------- ######
@pytest.fixture
def properties():
    prop = {'sampling_rate':4000, 'fft_win':5, 'fft_freq_range': [5, 121], 
            'fft_overlap':0.5, 'mains_noise': [58, 62]}
    return prop

@pytest.fixture
def fixed_frequency():
    def _method(index):
        freq = [5, 5 ,5, 5, 5, 5]
        return freq[index]
    return _method

@pytest.fixture
def frequency():
    def _method(index):
        freq = [5, 10 ,20, 60, 80, 100]
        return freq[index]
    return _method

@pytest.fixture
def amplitude():
    def _method(index):
        freq = [20, 10 ,15, 5, 6, 3]
        return freq[index]
    return _method

@pytest.fixture
def simple_sine(): 
    def _method(properties, frequency, amplitude, index):
        obj = SimSignal(properties, time_duration)        
        return obj.add_sines([frequency(index)], [amplitude(index)] )
    return _method

@pytest.fixture
def mains_noise_sine(): 
    def _method(properties, mains_noise_freq):
        rhythm = [15, 15]
        freq = [10, mains_noise_freq]
        freq_noise = [mains_noise_freq]
        amp = [5, 10]
        amp_noise = [3]
        obj = SimSignal(properties, time_duration)
        return  obj.add_sines_norm(freq, amp, rhythm) + obj.add_sines(freq_noise, amp_noise)
    return _method

@pytest.fixture
def mixed_sine(properties, frequency, amplitude, index):
   def _method(properties, frequency, amplitude, index):
        obj = SimSignal(properties, time_duration)        
        return obj.add_sines(frequency(index), amplitude(index))
   return _method


####--------------------------------------------------- ######


####---------------------------- Tests -------------------------- ######
@pytest.mark.parametrize("freq_vector, input_freqs", 
                          [(np.arange(0, 100, 0.2), [8, 'test']), 
                          (np.arange(0, 100, 2), [1, '2', 3]),
                                                ])
def test_get_freq_index(freq_vector, input_freqs):
    """
    Test if function raises exception for incorrect input types
    
    Parameters
    ----------
    freq_vector : np.ndarray, frequency vector to be indexed
    input_freqs : values to find the index
    test_freqs : ground truth values

    Returns
    -------
    None.

    """
    with pytest.raises(Exception):
        get_freq_index(freq_vector, input_freqs)


@pytest.mark.parametrize("freq_vector, input_freqs, test_freqs", 
                          [(np.arange(0, 100, 0.2), [8], [40]), 
                          (np.arange(0, 100, 2), [8], [4]),
                          (np.arange(3, 100, 3), [5], [1]),
                          (np.arange(4, 100, 0.2), [6], [10]),
                          (np.arange(2, 100, 1), [10, 42], [8, 40])
                                                ])
def test_get_freq_index_output(freq_vector, input_freqs, test_freqs):
    """
    Test if function raises exception for incorrect input types
    
    Parameters
    ----------
    freq_vector : np.ndarray, frequency vector to be indexed
    input_freqs : values to find the index
    test_freqs : ground truth values

    Returns
    -------
    None.

    """
    
    freq_idx = get_freq_index(freq_vector, input_freqs)
    
    assert np.all(test_freqs == freq_idx) 


@pytest.mark.parametrize("freq_range", [ ([1, 10 ,200]),
                                          ([1, '10']),
                                          ([-10, 100]),
                                          ([100, 10]),
                                          ([1, 3000]),
                                          ])
def test_freq_range(properties:dict, freq_range:list):
    """
    Test if function raises exception for incorrect input types

    Parameters
    ----------
    properties : dict, with properties
    freq_range : list, with tests for frequency range input
    """
    
    # pass inputs to properties dictionary
    properties.update({'freq_range': freq_range})
    
    with pytest.raises(Exception):
        # init stft object
        Stft(properties)

        
@pytest.mark.parametrize("mains_noise", [ ([1, 10 ,200]),
                                          ([1, '10']),
                                          ([3, 100]),
                                          ([100, 10]),
                                          ([1, 200]),
                                          ])
def test_mains_noise_range(properties:dict, mains_noise:list):
    """
    Test whether inappropriate mains noise range inputs raise exception

    Parameters
    ----------
    properties : dict, with properties
    freq_range : list, with tests for frequency range input
    """
    
    # pass inputs to properties dictionary
    properties.update({'mains_noise': mains_noise})
    
    with pytest.raises(Exception):
        # init stft object
        Stft(properties)


@pytest.mark.parametrize("index", [(0), (1), (2), (3), (4)])    
def test_stft_simple_sine_freq(simple_sine, properties, frequency, amplitude, index):
    """
    Test if correct peak frequency is detected from stft analysis

    Parameters
    ----------
    simple_sine : func, create sine wave
    properties : dict, with main properties
    frequency : func, get freqeuncy value based on index
    amplitude : func, get amplitude value based on index
    index : int, used to select freq and amplitude

    Returns
    -------
    None.

    """
    
    # init stft object
    stft_obj = Stft(properties)

    # get stft
    pmat = stft_obj.get_stft(simple_sine(properties, frequency, amplitude, index))
    
    # get psd
    psd = np.mean(pmat, axis = 1)
    
    # get peak frequency
    peak_freq = stft_obj.f[np.argmax(psd)]
    
    assert peak_freq - frequency(index) < 0.001


@pytest.mark.parametrize("index", [([0, 2]), ([3, 1]), ([5, 4]), ([1, 1])])    
def test_stft_simple_sine_amp(simple_sine, properties, fixed_frequency, amplitude, index):
    """
    Test if relative amplitude of the signal is translated to relative PSD power

    Parameters
    ----------
    simple_sine : func, create sine wave
    properties : dict, with main properties
    fixed_frequency : func, get fixed freqeuncy value from function
    amplitude : func, get amplitude value based on index
    index : int, used to select freq and amplitude

    Returns
    -------
    None.

    """
        
    power = []
    amp = []
    for i in index:
        # init stft object
        stft_obj = Stft(properties)
    
        # get stft
        pmat = stft_obj.get_stft(simple_sine(properties, fixed_frequency, amplitude, i))
        
        # get psd
        psd = np.mean(pmat, axis = 1)
        
        # get frequency index to find peak power
        freq_idx = get_freq_index(stft_obj.f, [fixed_frequency(i)])
        
        # get power
        power.append(psd[freq_idx])
        amp.append(amplitude(i))
    
    assert np.sign(power[0] - power[1]) == np.sign(amp[0] - amp[1])


def test_mains_noise(mains_noise_sine, properties):
    """
    Test mains noise removal from PSD

    Parameters
    ----------
    mains_noise_sine : func, create signal with mains noise
    properties : dict, with main properties

    Returns
    -------
    None.

    """
    mains_noise_freq = 60
    properties.update({'mains_noise':[mains_noise_freq - 1, mains_noise_freq + 1]})
    
    # init stft object
    stft_obj = Stft(properties)

    # get stft
    pmat = stft_obj.get_stft(mains_noise_sine(properties, mains_noise_freq))
    
    # get frequency index to find peak power
    freq_idx = get_freq_index(stft_obj.f, [mains_noise_freq, properties['mains_noise'][0]])
    
    # get power at noise frequency before noise removal
    psd = np.mean(pmat, axis = 1)
    power_before_noise_removal =  psd[freq_idx[0]]
    
    # remove mains noise
    pmat = stft_obj.remove_mains(stft_obj.f, pmat)
    
    # get power at noise frequency after noise removal
    psd = np.mean(pmat, axis = 1)
    power_after_noise_removal =  psd[freq_idx[0]]
    
    # get power of lower limit
    psd = np.mean(pmat, axis = 1)
    nearby_power =  psd[freq_idx[1]]
    
    # check if noise is reduced and same as lower bound
    test = np.zeros(2, dtype = bool)
    test[0] = power_before_noise_removal > power_after_noise_removal
    test[1] = nearby_power == power_after_noise_removal
    assert np.all(test)








































