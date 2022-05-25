########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from beartype import beartype
from typing import TypeVar
PandasDf = TypeVar('pandas.core.frame.DataFrame')
from processing.stft import get_freq_index
########## ---------------------------------------------------------------- ##########

@beartype
def get_power_area(pmat:np.ndarray, freq_vec:np.ndarray, freqs:np.ndarray, average=True) -> np.ndarray:
    """
    Get power area across frequencies.

    Parameters
    ----------
    pmat : np.ndarray, 2D array containing power values rows = frequency bins and cols = time bins
    freq_vec : np.ndarray, vector with real frequency values
    freqs : np.ndarray, 2D array with frequencies, rows = different frequency ranges, colums = [start, stop]
    average : bool, if True gets the average across frequency and time

    Returns
    -------
    powers : 1D np.array, len = frequency ranges

    """
    
    # get frequency index
    freqs = freqs.reshape([-1, 2]) # reshape to 2D
    freq_idx = get_freq_index(freq_vec, freqs)
    
    axis = None
    if not average:
        axis = 0
    
    # init empty array to store powers
    powers = []
    for i in range(freq_idx.shape[0]):
        powers.append(np.mean(pmat[freq_idx[i,0]:freq_idx[i,1],:], axis=axis))
    
    return np.array(powers)

@beartype
def get_power_ratio(pmat:np.ndarray, freq_vec:np.ndarray, freqs:np.ndarray) -> np.ndarray:
    """
    Get power ratio across frequencies.

    Parameters
    ----------
    pmat : np.ndarray, 2D array containing power values rows = frequency bins and cols = time bins
    freq_vec : np.ndarray, vector with real frequency values
    freqs : np.ndarray, 3D array with frequencies, 1d = frequency ratios, 2d [lower to upper freq range], 3d = [start, stop]

    Returns
    -------
    powers : 1D np.array, len = frequency ranges 

    """
    
    # get frequency index
    freqs = freqs.reshape([-1,2,2]) # reshape to 3D
    freq_idx = get_freq_index(freq_vec, freqs)
    
    # init empty array to store powers
    powers = np.zeros(freq_idx.shape[0])
    for i in range(freq_idx.shape[0]):
        powers[i] = np.divide(np.mean(pmat[freq_idx[i,0,0]:freq_idx[i,0,1],:]),
                              np.mean(pmat[freq_idx[i,1,0]:freq_idx[i,1,1],:]))                           
    return powers

@beartype
def melted_power_area(index_df:PandasDf, power_df:PandasDf, freqs:list, selected_categories:list):
    """
    Get power area and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with power area and categories

    """
    # convert to numpy array
    freqs = np.array(freqs)
    
    # create frequency column names
    freq_columns = []
    for i in range(freqs.shape[0]):
        freq_columns.append(str(freqs[i,0]) + ' - ' + str(freqs[i,1]) + ' Hz')
    
    # create array for storage
    power_array = np.empty((len(index_df), freqs.shape[0]))
    for i in range(len(index_df)): # iterate over dataframe
        
        # get power across frequencies
        power_array[i,:] = get_power_area(power_df['pmat'][i], power_df['freq'][i], freqs)
        
    # concatenate to array
    index_df = pd.concat([index_df, pd.DataFrame(data = power_array, columns = freq_columns)], axis=1)
    
    # set file id as index
    index_df['id'] = index_df['animal_id'].astype(str) + index_df['file_id'].astype(str)
    index_df.set_index('id', inplace = True)    
    
    # melt dataframe for seaborn plotting
    df = pd.melt(index_df, id_vars = selected_categories, value_vars = freq_columns, var_name = 'freq', value_name = 'power_area',
                 ignore_index = False)
    
    return df

@beartype
def melted_power_ratio(index_df:PandasDf, power_df:PandasDf, freqs:list, selected_categories:list):
    """
    Get power ratio and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with power area and categories

    """
    # convert to numpy array
    freqs = np.array(freqs)
    
    # create frequency column names
    freq_columns = []
    for i in range(freqs.shape[0]):
        freq_columns.append(str(freqs[i,0]) + ' - ' + str(freqs[i,1]) + ' Hz')
    
    # create array for storage
    power_array = np.empty((len(index_df), freqs.shape[0]))
    for i in range(len(index_df)): # iterate over dataframe
        
            # get power across frequencies
            power_array[i,:] = get_power_ratio(power_df['pmat'][i], power_df['freq'][i], freqs)
            
    # concatenate to array
    index_df = pd.concat([index_df, pd.DataFrame(data = power_array, columns = freq_columns)], axis=1)
    
    # set file id as index
    index_df['id'] = index_df['animal_id'].astype(str) + index_df['file_id'].astype(str)
    index_df.set_index('id', inplace = True)  
        
    # melt dataframe for seaborn plotting
    df = pd.melt(index_df, id_vars = selected_categories, value_vars = freq_columns, var_name = 'freq', value_name = 'power_ratio',
                 ignore_index=False)
    
    return df

# @beartype
def melted_power_time(index_df:PandasDf, power_df:PandasDf, freqs:list, 
                    selected_categories:list, win_rebin):
    """
    Get power area and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted
    rebin_factor : int, window size for rebinning
    
    Returns
    -------
    df : PandasDf, melted df with power area and categories

    """
    # convert to numpy array
    freqs = np.array(freqs)
    
    # create frequency column names
    freq_columns = []
    for i in range(freqs.shape[0]):
        freq_columns.append(str(freqs[i,0]) + ' - ' + str(freqs[i,1]) + ' Hz')
    
    # create array for storage
    resample_factor = int(win_rebin[1]/win_rebin[0])
    df_list = []
    for i in range(len(index_df)): # iterate over dataframe
        # get power across frequencies and time
        power_array = get_power_area(power_df['pmat'][i], power_df['freq'][i], freqs, average=False)

        # resample
        rows, cols = power_array.shape
        trim = cols - cols%resample_factor
        new_cols = cols//resample_factor
        power_array = np.mean(power_array[:,:trim].reshape((rows, new_cols, resample_factor)), axis=2)
        t = np.tile(np.arange(0, power_array.shape[1])*win_rebin[1], power_array.shape[0])
        
        # create df and pass to list
        df = pd.DataFrame({'power_area':power_array.flatten(), 
                                     'freq':np.repeat(freq_columns, power_array.shape[1]),
                                     'time': t})
        df.insert(0, 'Index', i)
        df_list.append(df)

    # concatenate to array
    df = pd.concat(df_list, ignore_index=True)
    df = index_df.join(df.set_index('Index'))
    
    # set file id as index
    df['id'] = df['animal_id'].astype(str) + df['file_id'].astype(str)
    df.set_index('id', inplace=True)
    
    return df[selected_categories + ['freq', 'time', 'power_area']]


def melted_psds(index_df:PandasDf, power_df:PandasDf, freq_range:list, selected_categories:list): ## don't drop file index
    """
    Get PSD and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with psd and categories

    """

    # create arrays for storage
    power_array = np.array([])
    freq_array = np.array([])
    repeat_array = np.zeros(len(index_df))
    
    # get selected columns
    index_df['id'] = index_df['animal_id'].astype(str) + index_df['file_id'].astype(str)
    df = index_df[['id'] + selected_categories]
    
    for i in range(len(index_df)): # iterate over dataframe
        
        # unpack frequency and power
        freq = power_df['freq'][i]
        power = power_df['pmat'][i]
        
        # get desired frequency index
        f_idx = get_freq_index(freq, freq_range)
        freq = freq[f_idx[0]:f_idx[1]+1]
        power = np.mean(power[f_idx[0]:f_idx[1]+1,:], axis =1)
        
        # append to array
        power_array = np.concatenate((power_array, power))
        freq_array = np.concatenate((freq_array, freq ))
        
        # get length
        repeat_array[i] = freq.shape[0]

    # repeat array
    df = df.reindex(df.index.repeat(repeat_array))
    
    # set file id as index
    df.set_index('id', inplace = True) 
    
    # append to dataframe
    df['freq'] = freq_array
    df['power'] = power_array

    return df

def melted_power_dist(index_df:PandasDf, power_df:PandasDf, freq_range:list, selected_categories:list): ## don't drop file index
    """
    Get Power distribution and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with pdf and categories

    """

    # create arrays for storage
    power_array = np.array([])
    density_array = np.array([])
    threshold_array = np.array([])
    repeat_array = np.zeros(len(index_df))
    
    # get selected columns
    index_df['id'] = index_df['animal_id'].astype(str) + index_df['file_id'].astype(str)
    df = index_df[['id'] + selected_categories]
    
    # get all power areas
    for i in range(len(index_df)): # iterate over dataframe
        
        # unpack frequency and power
        freq = power_df['freq'][i]
        power = power_df['pmat'][i]
        
        # get desired frequency index
        f_idx = get_freq_index(freq, freq_range)
        freq = freq[f_idx[0]:f_idx[1]+1]
        power =  np.mean(power[f_idx[0]:f_idx[1]+1,:], axis = 0)
        
        power_df.at[i, 'pmat'] = power
        
        # append to array
        power_array = np.concatenate((power_array, power))
    
    # get mean and sdev for normalization
    avg = np.mean(power_array)
    sdev = np.std(power_array)
    
    # define edges for z normalized data and preallocate power_array
    power_array = np.array([])
    edges = np.linspace(-5, 5, 100)
    for i in range(len(index_df)):    
        
        # normalize
        power = (power_df['pmat'][i] - avg )/ sdev
        
        # select power above threshold
        threshold =  np.mean(power) + np.std(power)
        
        # get kde
        pdf = gaussian_kde(power, bw_method = 1).evaluate(edges)
        
        # append to array
        power_array = np.concatenate((power_array, edges))
        density_array = np.concatenate((density_array, pdf))
        threshold_array = np.concatenate((threshold_array, np.repeat(threshold, edges.shape[0])))
        
        # get length
        repeat_array[i] = edges.shape[0]

    # repeat array
    df = df.reindex(df.index.repeat(repeat_array))
    
    # set file id as index
    df.set_index('id', inplace = True) 
    
    # append to dataframe
    df['threshold'] = threshold_array
    df['power'] = power_array
    df['density'] = density_array

    return df


def norm_power(index_df, power_df, selection):
    """
    Normalize power by PSDs of selected condition, drop non matching conditions

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    selection : List/Tuple, (column name, baseline group)

    Returns
    -------
    index_df : PandasDf, with dropped indices where conditions are missing
    power_df : PandasDf, with dropped indices where conditions are missing
    
    """
    unique_id = 'animal_id'
    category, group  = selection

    # get number unique entries
    unique_groups = index_df[category].unique()
    unique_regions = index_df['brain_region'].unique()
    unique_ids = index_df[unique_id].unique()
    
    # iterate over brain regions
    for region in unique_regions:

        # iterate over unique ids
        for uid in unique_ids:   

            # get matching idx
            matching_entries = index_df[(index_df[unique_id] == uid) & (index_df['brain_region'] == region)]
    
            # drop groups that are not complete
            if len(matching_entries) < len(unique_groups):
                power_df = power_df.drop(matching_entries.index, axis=0)
                index_df = index_df.drop(matching_entries.index, axis=0)
            else:
                # get baseline psd
                base_idx = matching_entries[matching_entries[category] == group].index[0]
                base_psd = np.mean(power_df.pmat[base_idx], axis=1)
                
                # divide matching groups by baseline psd
                for i in matching_entries.index:
                    power_df.at[i, 'pmat'] = power_df['pmat'][i] / base_psd[:,None]

    return index_df.reset_index().drop(['index'], axis=1), power_df.reset_index().drop(['index'], axis=1)



def norm_power_unpaired(index_df, power_df, selection):
    """
    Normalize power by PSDs of selected condition, drop non matching conditions

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    selection : List/Tuple, (column name, baseline group)

    Returns
    -------
    index_df : PandasDf, with dropped indices where conditions are missing
    power_df : PandasDf, with dropped indices where conditions are missing
    
    """

    unique_id = 'animal_id'
    category, group  = selection

    # get number unique entries
    unique_groups = index_df[category].unique()
    unique_regions = index_df['brain_region'].unique()
    unique_ids = index_df[unique_id].unique()
    
    # iterate over brain regions
    for region in unique_regions:

        # iterate over unique ids
        for uid in unique_ids:   

            # get matching idx
            matching_entries = index_df[(index_df[unique_id] == uid) & (index_df['brain_region'] == region)]
    
            # drop groups that are not complete
            if (matching_entries[category] == group).sum() == 0:
                power_df = power_df.drop(matching_entries.index, axis=0)
                index_df = index_df.drop(matching_entries.index, axis=0)
            else:
                # get baseline psd
                base_idx = matching_entries[matching_entries[category] == group].index[0]
                base_psd = np.mean(power_df.pmat[base_idx], axis=1)
                
                # divide matching groups by baseline psd
                for i in matching_entries.index:
                    power_df.at[i, 'pmat'] = power_df['pmat'][i] / base_psd[:,None]

    return index_df.reset_index().drop(['index'], axis=1), power_df.reset_index().drop(['index'], axis=1)


def norm_mean_power(power_df):
    """
    Normalize based on mean power.

    Parameters
    ----------
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df

    Returns
    -------
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df

    """
    power_df['pmat'] = power_df['pmat'].apply(lambda pmat: pmat/np.mean(pmat))

    return power_df
    

if __name__ == '__main__':
    
    import os, yaml
    from load_index import load_index
    # from facet_plot_gui import GridGraph
    
    ### ---------------------- USER INPUT -------------------------------- ###
    
    ## define path and conditions for filtering
    parent_folder = r'\\SUPERCOMPUTER2\Shared\acute_allo'

    ## define frequencies of interest
    with open('settings.yaml', 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
    
    ### ---------------------------------------------------------------- ####
    
    ## load data frame
    index_df = load_index(os.path.join(parent_folder, 'index.csv'))

    # get power
    # _, power_df = get_pmat(index_df, settings)    
    # power_df.to_pickle(os.path.join(parent_folder, 'power_mat.pickle'))
    
    # power_df = pd.read_pickle(os.path.join(parent_folder, 'power_mat_verified.pickle'))
    
    # normalize to baseline
    # index_df, power_df = norm_power(index_df, power_df, ['treatment', 'baseline1'])
    # df = melted_power_dist(index_df, power_df, [30,70], ['sex', 'treatment', 'brain_region'])
    
    # df = melted_power_ratio(index_df, power_df, settings['freq_ratios'], ['sex', 'treatment', 'brain_region']) #
    
    # import seaborn as sns
    
    # get melted power area
    # data = melted_power_area(index_df, power_df, settings['freq_ranges'], ['sex', 'treatment', 'brain_region'])
    # GridGraph(parent_folder, 'test.csv', data).draw_graph('box')
    
    # sns.catplot(data = df, x = 'freq', y = 'power_area', hue = 'treatment', col = 'sex', row = 'brain_region', kind = 'box')
    
    # # get melted psd
    # data = melted_psds(index_df, power_df, [1, 120], ['sex', 'treatment', 'brain_region'])
    # GridGraph(parent_folder,  'test.csv', data).draw_psd()

    # df.to_csv('melted_psd.csv',index=True)
    # path = r'C:\Users\panton01\Desktop\pydsp_analysis'
    # filename = 'power_area_df.csv'
    # df.to_csv(os.path.join(path, filename), index = False)
    
    # graph = GridGraph(path, filename)
    # graph.draw_graph('violin')











