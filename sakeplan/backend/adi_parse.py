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
from typing import Optional, Dict
### ------------------------------------###

class AdiParse:
    """
    Class to parse labchart files and retrieve information using the adi-reader library.
    """   
    
    @beartype
    def __init__(self, file_path:str, channel_structures:dict={}, block_index: Optional[int] = None):
        """
        Retrieve file properties and pass to self.properties

        Parameters
        ----------
        file_path : str
        channel_structures : dict, keys =  total channels, values = channel list
        block_index : int, optional. The default is None.

        Returns
        -------
        None.

        """

        # pass to object properties
        self.file_path = file_path
        self.file_name = os.path.basename(self.file_path)
        
        # Get block index and lengths
        adi_obj = self.read_labchart_file()           # read file
        block_len = np.zeros(adi_obj.n_records)       # create empty array for storage
        for block in range(adi_obj.n_records):
           block_len[block] = adi_obj.channels[0].n_samples[block] # get block length
       
        self.max_block = adi_obj.n_records - 1      # get max block number
        self.block_lengths = block_len              # get all block lengths
        if block_index is not None:
            self.set_block(block_index)               # set block if provided
        else:
            self.block = np.argmax(block_len)           # find the block with larger length
        
         
        # Get total number of channels
        self.n_channels = adi_obj.n_channels
        
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
    
    def set_block(self, block_index: int):
        """Validate & set current block."""
        if block_index is None:
            return
        if not (0 <= int(block_index) <= self.max_block):
            raise ValueError(f"Block {block_index} out of range 0..{self.max_block}")
        self.block = int(block_index)
    
    def get_channel_names(self):
        """
        Returns labchart names in a dataframe format.

        Returns
        -------
        df : pd.DataFrame
        """
        
        # read labchart file and create dataframe
        adi_obj = self.read_labchart_file()
        cols = ['channel_id', 'channel_name']
        df = pd.DataFrame(data=np.zeros((self.n_channels, len(cols))), columns=cols, dtype='object')
        
        # iterate over channels and get names
        for ch in range(self.n_channels):
            df.at[ch, 'channel_id'] = str(ch)
            df.at[ch, 'channel_name'] = adi_obj.channels[ch].name

        del adi_obj
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

        # read labchart file
        adi_obj = self.read_labchart_file()
        
        # get file length per channel and pass to df
        ch_length = [adi_obj.channels[int(x)].n_samples[self.block] for x in df['channel_id']]
        df['file_length'] = ch_length
        
        del adi_obj
        
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
        
        # read labchart file and retrieve comments
        adi_obj = self.read_labchart_file()
        comments = adi_obj.records[self.block].comments

        # add comments for each channel
        properties = {'text' : 'comment_text_', 'tick_position' : 'comment_time_'}

        # get channel order
        ch_idx = np.array([com.channel_ for com in comments])

        # iterate over properties
        for key, val in properties.items():
            
            # index array
            idx_array = np.array([getattr(com, key)for com in comments])
            
            # create temporary list and store comments per channel
            temp_coms = [] 
            for ch in range(self.n_channels): # iterate over channels
                temp_coms.append(idx_array[ (ch_idx == ch) | (ch_idx == -1) ]) #-1 means all channels
            
            # convert list to dataframe and add columns to df
            df_comments = pd.DataFrame(temp_coms)
            for i in range(df_comments.shape[1]): # iterate over max number of comments
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
        del adi_obj
        return df

            
    
    def get_all_file_properties(self):
        """
        Extracts file name, channel names, channel number, brain region, comment text and time.
        These information are added to a pandas DataFrame

        Returns
        -------
        df : pd.DataFrame

        """       
        
        # add properties to dataframe
        df = self.get_channel_names()
        df = self.add_file_name(df)
        df = self.add_comments(df)
        df = self.add_brain_region(df)
        df = self.add_file_length(df)
        df = self.add_block(df)
        df = self.add_sampling_rate(df)
         
        return df
    
    def get_block_coms(self) -> dict:
        """
        Build a single record for the “Block Selector Panel” DATA CONTRACT.

        Returns
        -------
        dict
            {
                "file_name": str,                     # e.g., "rat01_day1.adicht"
                "blocks_s": list[float],              # block lengths in seconds (one per block)
                "selected": int,                      # 0-based index of the selected/longest block
                "comments_by_block": dict[int, list[float]]
                    # For each block index b, a list of comment times (in seconds)
                    # RELATIVE to the start of block b.
                    # We include comments that are either global (channel_ == -1)
                    # or attached to the first channel (channel_ == 0), matching
                    # the prior behavior. Adjust the filter below if you want to
                    # aggregate across different channels.
            }

        Notes
        -----
        - All durations are returned in **seconds**.
        - Block lengths are derived from per-block sample counts and the file’s
        sampling rate.
        - Comment times are clipped to the duration of their respective block.
        """
        # Open file once
        adi_obj = self.read_labchart_file()

        # get blokc times in seconds
        fs =  int(1/adi_obj.channels[0].tick_dt[0]) # samples/sec (Hz)
        blocks_s = [float(np.round(n_samples / fs, 3)) for n_samples in self.block_lengths]
       
        # Iterate through each block/record
        comments_by_block: dict[int, list[float]] = {}
        for b in range(adi_obj.n_records):
            com_list = adi_obj.records[b].comments
            com_times = [x.time for x in com_list]
            comments_by_block[b] = com_times
        
        # create record
        record = {
            "file_name": self.file_name,
            "blocks_s": blocks_s,
            "selected": int(self.block),
            "comments_by_block": comments_by_block,
        }

        del adi_obj  # free resources
        return record

    
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
        
      
            
            
            
            
            
            
            
            
            
            
            
            
            
            