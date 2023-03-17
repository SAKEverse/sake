
### ----------------- IMPORTS ----------------- ###
import os
from beartype import beartype
import numpy as np
import pandas as pd
from backend.adi_parse import AdiParse
from backend.file_check import check_file
from backend import search_function
from backend.get_all_comments import GetComments
### ------------------------------------------- ###

@beartype
def get_file_data(folder_path:str, channel_structures:dict):
    """
    Get file data in dataframe

    Parameters
    ----------
    folder_path : str
    channel_structures : dict, keys =  total channels, values = channel list

    Returns
    -------
    file_data : pd.DataFrame

    """
    
    # make lower string and path type
    folder_path = os.path.normpath(folder_path.lower())
    file_data = pd.DataFrame()
    cntr = 0
    
    # walk through all folders
    for root, dirs, files in os.walk(folder_path):
        
        # get labchart file list
        filelist = list(filter(lambda k: '.adicht' in k, files))

        for file in filelist: # iterate over list
        
            # initiate adi parse object      
            adi_parse = AdiParse(os.path.join(root, file), channel_structures)
            
            # get all file data in dataframe
            temp_file_data = adi_parse.get_all_file_properties()
            
            # add folder path
            temp_file_data['folder_path'] = os.path.normcase(root)

            # apppend to dataframe
            file_data = file_data.append(temp_file_data, ignore_index = True)

            cntr+=1
                
    # convert data frame to lower case
    file_data = file_data.apply(lambda x: x.astype(str).str.lower())
        
    # convert file length to int
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
    animal_id = user_data[drop_idx].reset_index().drop(['index'], axis = 1)
    
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

    return file_data, user_data.drop(np.where(drop_idx)[0], axis = 0)


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


def get_index_array(folder_path, user_data):
    """
    Get file data, channel array and create index
    for experiments according to user selection

    Parameters
    ----------
    file_data : pd.DataFrame, aggregated data from labchart files
    user_data : 2D list, user search and grouping parameters from datatable

    Returns
    -------
    index_df: pd.DataFrame, with index
    group_columns: list, column names that denote groups
    warning_str: str, string used for warning

    """
    
    # get dataframe and convert to lower case
    user_data = pd.DataFrame(user_data)
    user_data = user_data.apply(lambda x: x.astype(str).str.lower())
  
    # remove rows with missing inputs
    user_data = user_data.dropna(axis = 0)

    warning_str = ''
    # # ensure group names are unique
    # if len(user_data['Assigned Group Name']) != len(user_data['Assigned Group Name'].unique()):
    #    warning_str += 'Duplicate -Assigned Group Names- were found. Please check that -Assigned Group Names- are unique'

    # get channel order
    channel_structures = get_channel_structures(user_data)
    
    # get all file data in dataframe
    check_file(folder_path, channel_structures)
    file_data = get_file_data(folder_path, channel_structures)
    
    # add animal id
    file_data, user_data = add_animal_id(file_data, user_data)
    
    # get index dataframe 
    index_df, group_columns, warning_add = create_index_array(file_data, user_data)
    warning_str += warning_add
    
    # check if no conditions were found
    if len(list(index_df.columns[index_df.columns.get_loc('stop_time')+1:])) < 2:
        warning_str += 'Warning: Only Brain region column was found!!'
        
    # check if multiple blocks are found
    if np.sum(index_df.block.astype(int)) > 0:
        warning_str += 'Warning: Some files contain more tha one block!!'
           
    return index_df, group_columns, warning_str

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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    