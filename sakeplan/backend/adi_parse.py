# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:31:17 2021

@author: panton01
"""

### ----------- IMPORTS --------------- ###
import os
from beartype import beartype
import numpy as np
import pandas as pd
import adi
### ------------------------------------###

class AdiParse:
    """
    Class to parse labchart files and retrieve information using the adi-reader library.
    """   
    
    @beartype
    def __init__(self, file_path:str, channel_structures:dict = {}):
        """
        Retrieve file properties and pass to self.properties

        Parameters
        ----------
        file_path : str
        channel_structures : dict, keys =  total channels, values = channel list

        Returns
        -------
        None.

        """

        
        # pass to object properties
        self.file_path = file_path
        
        # get file name
        self.file_name = os.path.basename(self.file_path)
        
        # Get block
        adi_obj = self.read_labchart_file()           # read file
        
        block_len = np.zeros(adi_obj.n_records)       # create empty array for storage
        for block in range(adi_obj.n_records):
            
            # get block length
           block_len[block] = adi_obj.channels[0].n_samples[block]
       
        self.block = np.argmax(block_len)           # find the block with larger length
         
        # Get total number of channels
        self.n_channels = adi_obj.n_channels
        
        # Get file length
        self.file_length = int(block_len[self.block])
        
        # get channel order if total channel number matches
        channel_order = []
        if self.n_channels in channel_structures.keys():
            channel_order = channel_structures[self.n_channels]
            
        # get channel order
        if len(channel_order) == 0:
            self.channel_order = 'Brain regions were not found'
        elif self.n_channels%len(channel_order) != 0:
            self.channel_order = 'Brain regions provided do not match channel order'
        else:
            self.channel_order = channel_order
            
        del adi_obj                                   # clear memory
    
    
    def read_labchart_file(self):
        """
        Creates read object for labchart file and passes to self

        Returns
        -------
        adi_obj : Labchart read object

        """
        
        # return labchart read object
        return adi.read_file(self.file_path)
    
    
    def get_channel_names(self):
        """
        Returns labchart names in a dataframe format.

        Returns
        -------
        df : pd.DataFrame

        """
        
        # read labchart file
        adi_obj = self.read_labchart_file()

        # define dataframe columns
        cols = {'channel_id', 'channel_name'}

        # create empty dataframe
        df = pd.DataFrame(data = np.zeros((self.n_channels, len(cols))), columns = cols, dtype = 'object')
        
        # iterate over channels and get names
        for ch in range(self.n_channels):
            
            # append channel info to dictionary
            df.at[ch, 'channel_id'] = str(ch)
            df.at[ch, 'channel_name'] = adi_obj.channels[ch].name

        del adi_obj # clear memory

        return df
    
    def add_file_name(self, df):
        """
        Adds file name to channels dataframe

        Parameters
        ----------
        df : pd.DataFrame


        Returns
        -------
        df : pd.DataFrame

        """
        
        # add file name
        df['file_name'] = self.file_name
        
        return df
    
    
    def add_block(self, df):
        """
        Adds block number to channels dataframe

        Parameters
        ----------
        df : pd.DataFrame


        Returns
        -------
        df : pd.DataFrame

        """
        
        # add file name
        df['block'] = self.block
        
        return df
    
    def add_file_length(self, df):
        """
        Adds file length (in samples) to channels dataframe

        Parameters
        ----------
        df : pd.DataFrame


        Returns
        -------
        df : pd.DataFrame

        """
        
        # add file length
        df['file_length'] = int(self.file_length)
        
        return df
    
    def add_comments(self, df):
        """
        Add one column to a channels df for each comment and comment time across channels.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        df : pd.DataFrame

        """
        
        # read labchart file
        adi_obj = self.read_labchart_file()
        
        # add comments for each channel
        properties = {'text' : 'comment_text_', 'tick_position' : 'comment_time_'}
        
        # retrieve all comments
        comments = adi_obj.records[0].comments;
        
        # get channel order
        ch_idx = np.array([com.channel_ for com in comments])

        # iterate over properties
        for key, val in properties.items():
            
            # index array
            idx_array = np.array([getattr(com, key)for com in comments])
            
            temp_coms = [] # creaty empty list
            for ch in range(self.n_channels): # iterate over channels
                
                # get attribute that belongs in a channel
                temp_coms.append(idx_array[ (ch_idx == ch) | (ch_idx == -1) ]) #-1 means all channels
            
            # convert list to dataframe
            df_comments = pd.DataFrame(temp_coms)
            
            # add columns to dataframe
            for i in range(df_comments.shape[1]): # iterate over max number of comments
                
                # pass to dataframe
                df[val + str(i)] = df_comments.iloc[:,i]
                
        del adi_obj # clear memory
        
        return df
    
    def add_brain_region(self, df):
        """
        Add brain regions to channels dataframe

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        df : pd.DataFrame

        """
        
        if type(self.channel_order) == str:
            df['brain_region'] = self.channel_order
        else:
            df['brain_region'] = self.channel_order * int(self.n_channels/len(self.channel_order))
            
        return df
    
    def add_sampling_rate(self, df):
        """
        Add sampling rate to channels in dataframe

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        df : pd.DataFrame

        """
        
        # read labchart file
        adi_obj = self.read_labchart_file()
        
        # get sampling rate
        for i in range(len(df)):
            df.at[i,'sampling_rate'] = int(1/adi_obj.channels[i].tick_dt[self.block])
            
        return df
            
    
    def get_all_file_properties(self):
        """
        Extracts file name, channel names, channel number, brain region, comment text and time.
        These information are added to a pandas DataFrame

        Returns
        -------
        df : pd.DataFrame

        """       
        
        # add channel names
        df = self.get_channel_names()
        
        # add file names
        df = self.add_file_name(df)
        
        # add comments
        df = self.add_comments(df)
        
        # add brain region
        df = self.add_brain_region(df)
        
        # add file length
        df = self.add_file_length(df)
        
        # add file length
        df = self.add_block(df)
        
        # add sampling rate
        df = self.add_sampling_rate(df)
         
        return df
            
        
    
    def get_unique_conditions(self):
        """
        Get unique conditions from channel names.
        
        For example
        ----------
        ch1 = m_wt_cus
        ch2 = f_ko_cussage
        ch3 = m_ko_cus
        
        unique_groups = [[m, f], [wt, ko], [cus, cussage]]

        Raises
        ------
        FileNotFoundError

        Returns
        -------
        df : Pd. DataFrame of separated groups per channel
        unique_groups : List of lists with unique groups

        """
        
        # get channel names
        df = self.get_channel_names()
        
        # create empty list to store length of each condition
        condition_list = []
        
        for i,name in enumerate(df.channel_name): # iterate over channels
            
            # get list of conditions
            condition_list.append(name.split('_'))
         
        try:
            # convert to pandas dataframe
            df = pd.DataFrame(condition_list)
        except Exception as err:
            raise FileNotFoundError(f'Unable to read conditions.\n{err}')
        
        # get unique groups for each condition
        unique_groups = [df[column].unique().tolist() for column in df]
        
        return df, unique_groups
            
          
    @beartype
    def filter_names(self, filters:str):
        """
        Filters channels names and returns index

        Parameters
        ----------
        filters : str

        Returns
        -------
        idx : List : List with indices of channels that contain filter

        """
        
        # get channel names
        df = self.get_channel_names()
        
        # get index of channels that contain filter
        idx = df.index[df.channel_name.str.contains(filters, case=False)].to_list()
        return idx
            
    # create unique sets of attributes from channel names to autopopulate the user drop down menus        


if __name__ == '__main__':
    
    file_path = r'C:\Users\panton01\Desktop\example_files\#1animal_sdsda_wt.adicht'#r'Z:\Pantelis\Combined Cohorts 1 & 2 (4 Channel)\PV-dlx-hm3d_d1(3)_1-4_comments.adicht'
    channel_order = ['na', 'Bla', 'Hipp', 'Emg']
    # initiate object      
    adi_parse= AdiParse(file_path, channel_order)
    
    adi_obj = adi_parse.read_labchart_file()
    
    df =  adi_parse.get_all_file_properties()
    

    
    # df, unique_groups = adi_parse.get_unique_conditions()

    # df.to_csv('sankey_data.csv')
    #
    # idx = adi_parse.filter_names('fc')
      
            
            
            
            
            
            
            
            
            
            
            
            
            
            