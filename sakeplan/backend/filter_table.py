### ----------------- IMPORTS ----------------- ###
import os
from beartype import beartype
import numpy as np
import pandas as pd
from backend.adi_parse import AdiParse
from backend.file_check import check_file
from backend import search_function
from backend.get_all_comments import GetComments
from typing import Optional, Dict
### ------------------------------------------- ###

@beartype
def get_file_data(folder_path: str, channel_structures: dict, selected_blocks: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Get file data as a dataframe. If `selected_blocks` is provided, use that block
    per file instead of the default block from AdiParse.

    Parameters
    ----------
    folder_path : str
    channel_structures : dict   # {total_channels: [region1, region2, ...]}
    selected_blocks : dict | None
        Optional mapping {file_name: block_index} (file_name = basename like 'file1.adicht').

    Returns
    -------
    file_data : pd.DataFrame
    """
    # normalize inputs
    folder_path = os.path.normpath(folder_path.lower())

    # make sure selected_blocks is in correct format
    selected_blocks = { (k or "").lower(): int(v) for k, v in (selected_blocks or {}).items()}

    # iterate over files and collect data
    frames = []
    for root, dirs, files in os.walk(folder_path):
        filelist = [f for f in files if f.endswith('.adicht')]
        for file in filelist:
            
            # get info from labchart file
            fpath = os.path.join(root, file)
            sb = selected_blocks.get(file.lower(), None)
            adi_parse = AdiParse(fpath, channel_structures, block_index=sb)
            temp = adi_parse.get_all_file_properties()

            # add folder path column
            temp['folder_path'] = os.path.normcase(root)
            frames.append(temp)

    # concat + normalize
    file_data = pd.concat(frames).reset_index(drop=True)
    file_data = file_data.apply(lambda x: x.astype(str).str.lower())

    # ensure types
    file_data['file_length'] = file_data['file_length'].astype(np.int64)

    # make paths relative
    file_data.folder_path = file_data.folder_path.str.replace(folder_path, '', regex=False)
    file_data.folder_path = file_data.folder_path.map(lambda x: x.lstrip('\\'))

    return file_data

def get_channel_structures(user_data):
    """
    Get channel structure from labchart files based on user data

    Parameters
    ----------
    user_data : Dataframe with user data for SAKE input

    Returns
    -------
    order : List with channels in order
    """
    
    # define separator
    separtor = '-'
    
    # get data containing channel order
    channel_structures = user_data[user_data['Source'] == 'total_channels'].reset_index().drop(['index'], axis = 1)
    
    regions = {}
    for i in range(len(channel_structures)):
        
        # retrieve channel names
        channel_names = channel_structures['Assigned Group Name'][i]
        
        # get list of channels for each total channels entry
        region_list = channel_names.split(separtor)
        regions.update({int(channel_structures['Search Value'][i]): region_list})
        
    return regions

def add_animal_id(file_data, user_data):
    """
    Add animal id from channel name to labchart data

    Parameters
    ----------
    file_data : pd.DataFrame
    user_data : Dataframe with user data for SAKE input

    Returns
    -------
    file_data : List with channels in order
    user_data: Dataframe with user data for SAKE input

    """
    
    # get data containing channel order
    drop_idx = user_data['Search Function'] == 'within'
    animal_id = user_data[drop_idx].reset_index().drop(['index'], axis=1)
    
    # check if present
    if len(animal_id) > 1:
        raise(Exception('Only one Search Function with -within- is allowed!\n'))
    if len(animal_id)  == 0:
        raise(Exception('Search Function -within- is required!\n'))
    
    # convert to dictionary
    ids = animal_id.loc[0].to_dict()
    
    # define separator
    sep = ids['Search Value']
    
    # get file name
    # ids['Category']
    file_data['animal_id'] = ''
    for i,name in enumerate(file_data[ids['Source']]):
        if sep in name:
            file_data.at[i, ids['Category']] = sep + name.split(sep)[1] + sep

    return file_data, user_data.drop(np.where(drop_idx)[0], axis=0)


def get_categories(user_data):
    """
    Get unique categories and groups in dictionary.

    Parameters
    ----------
    user_data : pd.DataFrame, with user group inputs.

    Returns
    -------
    groups : dict, keys are unique categories and groups.
    
    """
    
    # get unique categories
    unique_categories = user_data['Category'].unique()
    
    groups = {} # create group dictionary
    for category in unique_categories: # iterate over categories
        
        # which groups exist in categories
        groups.update({category: list(user_data['Assigned Group Name'][user_data['Category'] == category]) })
        
    return groups


def reverse_hot_encoding(sort_df):
    """
    Reverse hot coding in dataframe and replace with column names or nan
    

    Parameters
    ----------
    sort_df : pd.DataFrame, with columns in one hot encoding format

    Returns
    -------
    col_labels: 1D np.array with columns retrieved from one hot encoded format

    """

    # get columns
    labels = np.array(sort_df.columns)
    
    # find index where column is True #np.argmax(np.array(sort_df), axis = 1)
    idx_array = np.array(sort_df)
    col_labels = np.zeros(len(sort_df), dtype=object)
    
    for i in range(idx_array.shape[0]): # iterate over idx_array
        
        # find which column
        idx = np.where(idx_array[i] == True)[0]
        
        if len(idx) == 0:       # if no True value present
            col_labels[i] = np.NaN
        elif  len(idx) > 1:     # if more than one True value present
            col_labels[i] = np.NaN
        elif len(idx) == 1:     # if one True value present
            col_labels[i] = labels[idx[0]]
            
    return col_labels
    

def convert_logicdf_to_groups(index_df, logic_index_df, groups_ids:dict):
    """
    Convert logic from logic_index_df to groups and and append to index_df

    Parameters
    ----------
    index_df : pd.DataFrame, to append categories
    logic_index_df : pd.DataFrame, containing logic
    groups_ids : dict, containg categories as keys and groups as values

    Returns
    -------
    index_df : pd.DataFrame

    """
    
    # convert logic to groups
    for category, groups in groups_ids.items():
        
        # check if all groups present in dataframe
        groups_present = all(elem in logic_index_df.columns for elem in groups)
        
        if (groups_present == True): # are all groups present in dataframe?
            if (logic_index_df[groups].any().any() == True):  # was any group detected? 
            
                # convert logic to groups
                index_df[category] = reverse_hot_encoding(logic_index_df[groups])
                
    return index_df

    
def get_source_logic(file_data, user_data, source:str):
    """
    Find which unique groups exist and return as dataframe

    Parameters
    ----------
    user_data : pd.DataFrame
    source : str, source destination

    Returns
    -------
    index : pd.DataFrame

    """
    
    # get only user data form source
    user_data = user_data[user_data['Source'] == source].reset_index()
    
    index = {}
    for i in range(len(user_data)): # iterate over user data entries       

        # find index for specified source and match string
        idx = getattr(search_function, user_data.at[i, 'Search Function'])(file_data[source], user_data.at[i, 'Search Value'])                  
        
        # append to index dictionary                
        index.update({user_data.at[i, 'Assigned Group Name']: idx})
        
    return pd.DataFrame(index)


def get_drop_logic(file_data, user_data, source:str):
    """
    Find which unique groups exist and return as dataframe

    Parameters
    ----------
    user_data : pd.DataFrame
    source : str, source destination

    Returns
    -------
    index : pd.DataFrame

    """

    # get only user data form source
    user_data = user_data[user_data['Source'] == source].reset_index()
   
    index = {}
    for i in range(len(user_data)): # iterate over user data entries       
                
        # find index for specified source and match string
        idx = getattr(search_function, user_data.at[i, 'Search Function'])(file_data[source], user_data.at[i, 'Search Value'])

        # append to index dictionary
        col_name =  source + '_' + user_data.at[i, 'Assigned Group Name'] + str(i)
        index.update({col_name: idx})
        
    return pd.DataFrame(index)
    

def create_index_array(file_data, user_data):
    """
    Create index for experiments according to user selection

    Parameters
    ----------
    file_data : pd.DataFrame, aggregated data from labchart files
    user_data : pd.DataFrame, user search and grouping parameters

    Returns
    -------
    index_df: pd.DataFrame, with index
    group_columns: list, column names that denote groups
    warning_str: str, string used for warning
    """

    # create empty dataframes for storage
    logic_index_df = pd.DataFrame()
    index_df = pd.DataFrame()
    drop_df = pd.DataFrame()
    warning_str = ''
    
    # create sources list
    sources = ['channel_name', 'file_name']
    
    # separate user data based on drop
    drop_idx = user_data['Assigned Group Name'] == 'drop'
    user_data_drop = user_data[drop_idx]
    user_data_use = user_data[~drop_idx]
    
    for source in sources: # iterate over user data entries  
        
        # get index logic for each assigned group
        df = get_source_logic(file_data, user_data_use, source)
        logic_index_df = pd.concat([logic_index_df, df], axis=1)
        
        # get drop_logic
        df = get_drop_logic(file_data, user_data_drop, source)
        drop_df = pd.concat([drop_df, df], axis=1)

    # add columns from file to data
    add_columns = ['animal_id','folder_path','file_name','file_length',
     'channel_id', 'block' , 'sampling_rate', 'brain_region',]
    index_df = pd.concat([index_df, file_data[add_columns]], axis=1)
    
    # get time
    index_df['start_time'] = 1
    index_df['stop_time'] = file_data['file_length']

    # get category with group names
    groups_ids = get_categories(user_data_use)
    
    # convert logic to groups
    index_df = convert_logicdf_to_groups(index_df, logic_index_df, groups_ids)
    
    # remove rows containing drop
    region_drop = pd.DataFrame((index_df['brain_region'] == 'drop').rename('drop'))
    drop_df = pd.concat((drop_df, region_drop), axis=1)
    index_df = index_df[~drop_df.any(axis=1).values]
    file_data = file_data [~drop_df.any(axis=1).values]
    
    # get time and comments
    obj = GetComments(file_data, user_data_use, 'comment_text', 'comment_time')
    index_df, com_warning = obj.add_comments_to_index(index_df)
    
    # check if user selected time exceeds bounds
    if (index_df['start_time']<0).any() or (index_df['start_time']>index_df['file_length']).any():
        raise Exception('Start time exceeds bounds.')
    elif (index_df['stop_time']<0).any() or (index_df['stop_time']>index_df['file_length']).any():
        raise Exception('Stop time exceeds bounds.')

    # update group columns
    group_columns = list(index_df.columns[index_df.columns.get_loc('stop_time')+1:]) + ['brain_region']
    
    # check if groups were not detected
    if index_df.isnull().values.any():
        warning_str = 'Warning: Some conditons were not found!!'
    
    # put categories at end
    index_df = index_df[ [x for x in list(index_df.columns) if x not in group_columns] + group_columns]           
    return index_df, group_columns, warning_str + com_warning


def get_index_array(folder_path, user_data, file_data=None):
    """
    Get index array for experiments according to user selection

    Parameters
    ----------
    folder_path : str
        Path to folder containing labchart files
    user_data : pd.DataFrame
        With user search and grouping parameters
    file_data : pd.DataFrame or None, optional
        If provided, this dataframe is used instead of calling get_file_data().
        Must already have block selection applied.

    Returns
    -------
    index_df : pd.DataFrame
        Index dataframe
    group_columns : list
        Column names that denote groups
    warning_str : str
        String used for warning
    """

    # normalize user_data
    user_data = pd.DataFrame(user_data)
    user_data = user_data.apply(lambda x: x.astype(str).str.lower())
    user_data = user_data.dropna(axis=0)

    # initialize warning string
    warning_str = ''

    # get channel order
    channel_structures = get_channel_structures(user_data)

    # get or use file_data
    if file_data is None:
        check_file(folder_path, channel_structures)
        file_data = get_file_data(folder_path, channel_structures)

    # add animal id
    file_data, user_data = add_animal_id(file_data, user_data)

    # get index dataframe
    index_df, group_columns, warning_add = create_index_array(file_data, user_data)
    warning_str += warning_add

    # sanity check
    if len(list(index_df.columns[index_df.columns.get_loc('stop_time')+1:])) < 2:
        warning_str += 'Warning: Only Brain region column was found!!'

    return index_df, group_columns, warning_str

def get_index_df(folder_path, user_data, selection):
    """
    Get index dataframe for experiments according to user selection.

    Parameters
    ----------
    folder_path : str
        Path to folder containing labchart files
    user_data : list of dicts
        User data from SAKE input table
    selection : dict
        Mapping {file_name: block_index} (file_name = basename like 'file1.adicht').

    Returns
    -------
    index_df : pd.DataFrame
        Index dataframe
    group_columns : list
        Column names that denote groups
    warning_str : str
        String used for warning
    """

    # Normalize user table
    user_df = pd.DataFrame(user_data)
    user_df = user_df.apply(lambda x: x.astype(str).str.lower())
    user_df = user_df[user_df['Source'] != '']

    # Build channel structures
    channel_structures = get_channel_structures(user_df)

    # Map selected blocks → keys must be basenames + lower (backend expects that)
    # blocksel_store looks like {"file1.adicht": 0, "file2.adicht": 2, ...}
    selected_blocks = {}
    if selection:
        for k, v in selection.items():
            if k:
                selected_blocks[os.path.basename(k).lower()] = int(v)

    # Rebuild file_data with the chosen blocks
    file_data = get_file_data(folder_path, channel_structures, selected_blocks=selected_blocks)

    # Compute index_df using the selected blocks
    try:
        index_df, group_names, warning_str = get_index_array(folder_path, user_df, file_data=file_data)
    except TypeError:
        raise Exception("Could not create index array based on selected blocks.")

    return index_df, group_names, warning_str


if __name__ == '__main__':
    
    # define path
    folder_path = r'C:\Users\panton01\Desktop\example_files'
    
    # get user table data example
    user_data = pd.read_csv(r'C:\Users\panton01\Desktop\pydsp_analysis\user_data.csv')
    
    # convert data frame to lower case
    user_data = user_data.apply(lambda x: x.astype(str).str.lower())
    
    # remove rows with no source
    user_data = user_data[user_data.Source != '']
    
    # get channel order
    channel_structures = get_channel_structures(user_data)
    
    # get all file data in dataframe
    file_data = get_file_data(folder_path, channel_structures)

    # get experiment index
    index_df, group_columns, warning_str = create_index_array(file_data, user_data)

















