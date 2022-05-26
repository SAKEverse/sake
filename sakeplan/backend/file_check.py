# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
from backend.adi_parse import AdiParse
##### ------------------------------------------------------------------- #####

def check_file(folder_path:str, channel_structures:dict):
    """
    Get file data in dataframe

    Parameters
    ----------
    folder_path : str
    channel_structures : dict, keys =  total channels, values = channel list
    """
    
    # make lower string and path type
    folder_path = os.path.normpath(folder_path.lower())
    
    # walk through all folders
    for root, dirs, files in os.walk(folder_path):
        
        # get labchart file list
        filelist = list(filter(lambda k: '.adicht' in k, files))

        for file in filelist: # iterate over list
            
            try:
                # initiate adi parse object
                adi_parse = AdiParse(os.path.join(root, file), channel_structures)
                
                # get all file data in dataframe
                adi_parse.get_all_file_properties()
                
            except:
                raise Exception(file + ' could not be read.')
                   
    return



