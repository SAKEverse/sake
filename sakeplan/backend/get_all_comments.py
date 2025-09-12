### --------- IMPORTS --------- ###
import numpy as np
import pandas as pd
from backend import search_function
### --------------------------- ###


class GetComments:
    """
    Get experiment index from user comments and groups
    """
    
    @staticmethod
    def get_index_file_com(com_logic_df, time_array, label):
        """
        Add Nans to comments not present and get respective time.
        
        Parameters
        ----------
        com_logic_df : pd.DataFrame
        time_array : pd.DataFrame
        label : str
    
        Returns
        -------
        com_label : nd array,
        com_time : nd array,
    
        """
       
        # find index where column is True
        idx_array = np.array(com_logic_df)
        com_label = np.zeros(len(com_logic_df), dtype = bool)
        com_time = np.zeros(len(com_logic_df))
        
        for i in range(idx_array.shape[0]): # iterate over idx_array
            
            # find which column
            idx = np.where(idx_array[i] == True)[0]
            
            if len(idx) == 0:       # if no True value present
                com_label[i] = False
                com_time[i] = np.NaN
            else:
                com_label[i] = True
                com_time[i] = time_array.iloc[i, idx[0]]
    
        return com_label, com_time
    
    
    def __init__(self, file_data, user_data, comment_text:str, comment_time:str):
        """
        

        Parameters
        ----------
        file_data : pd.DataFrame, aggregated data from labchart files
        user_data : pd.DataFrame, user search and grouping parameters
        comment_text : str, to search file_data columns for comment text
        comment_time : str, to search file_data columns for comment times

        Returns
        -------
        None.

        """
        
        # pass to object
        self.comment_text = comment_text
        self.comment_time = comment_time

        # Get matching comment text and time columns from file_data dataframe
        self.com_text_cols = list(file_data.columns[file_data.columns.str.contains(self.comment_text)])
        self.com_time_cols = list(file_data.columns[file_data.columns.str.contains(self.comment_time)])
        self.com_match = dict(zip(self.com_text_cols, self.com_time_cols))
        
        # pass file data to object
        self.file_data = file_data    
        
        # get user data containing comment text
        self.user_data = user_data[user_data['Source'] == self.comment_text].reset_index(drop=True)
        
        # check if user data contain comments as source
        if len(self.user_data) == 0:    
            self.category = None # if not set category to none
        else:
        
            # get groups
            categories = list(self.user_data.Category.unique())
            
            # Get unique category
            if len(categories) > 1:
                raise Exception('Only one comment category is allowed')    
            self.category = categories[0]
       
    
    def get_index_per_comment(self, index:int):
        """
        Get boolean index for comment. True if a comment is present. 
        Along with it's respective time that occured.

        Parameters
        ----------
        index : int, index of self.user_data

        Returns
        -------
        comtext : np.array, true if comment present
        comtime : np.array, time of comment

        """
        
        # create empty arrays
        comtext = np.zeros( (len(self.file_data), len(self.com_match)) )
        comtime = np.zeros( (len(self.file_data), len(self.com_match)) )
        
        for i,text in enumerate(self.com_match):      # iterate over file comment columns
           
            # find index for specified source and match string
            idx = getattr(search_function, self.user_data.at[index, 'Search Function'],
                          )(self.file_data[text], self.user_data.at[index, 'Search Value'])
            
            # add to numpy arrays
            comtext[:,i] =  idx             
            comtime[:,i] =  self.file_data[self.com_match[text]]
            
        return comtext, comtime
     

    def get_comments_with_time(self, index_df, com_names, user_times, com_logic, com_time):
        """
        Convert comment logic to group names with their time to index_df
        
        Parameters
        ----------
        index_df : pd.DataFrame, containing experiment index
        com_names: list, with comment names
        user_times: list, with 2 element numpy arrays containing time selection  
        com_logic : np.array, containing detected comment logic
        com_time : np.array, containing time for each comment

        Returns
        -------
        index_df : pd.DataFrame, containing experiment + comment index 

        """
        
        # create empty array for one category
        category_df = pd.DataFrame(columns = list(index_df.columns) + [self.category])
       
        for i,comment in enumerate(com_names): # iterate over comments in one category
            
            # create temporary dataframe column for one category
            temp_df = index_df.copy()
            
            # get index where comment is present
            com_idx = com_logic[:,i]
            
            # reset index for new comment type
            if (i % len(self.com_match)) == 0:
                comment_suffix = np.zeros(com_idx.shape[0], dtype=int)
 
            # get categories where comment is present  
            if com_idx.any() == 1:
                comment_suffix += com_idx.astype(int)
                comment_name = np.array([comment + str(x) for x in comment_suffix])
                temp_df.at[com_idx, self.category] = comment_name[com_idx]
               
            # get times from comment
            fs = np.array(index_df['sampling_rate'], dtype=float)                       # convert sampling rate form string to float
            temp_df['start_time'] = com_time[:, i] + (user_times[i][0] * fs)            # get start time
            temp_df['stop_time'] = com_time[:, i] + (user_times[i][1] * fs)             # get stop time
            
            # concatenate to category_df
            category_df = pd.concat([category_df, temp_df], axis=0)

        # drop rows not containing the comments
        category_df = category_df[category_df[self.category].notna()]
        
        # convert times to int
        category_df['start_time'] = category_df['start_time'].astype(np.int64)
        category_df['stop_time'] = category_df['stop_time'].astype(np.int64)
        
        return  category_df
    

    
    def add_comments_to_index(self, index_df):
        """
        Adds comment to index_df (if exist).

        Parameters
        ----------
        index_df : pd.DataFrame, with experiment index

        Raises
        ------
        Exception

        Returns
        -------
        index_df : pd.DataFrame, with experiment index (+ comments)
        com_warning: str, warning when comments are not found in a channel

        """
        # init comment warning
        com_warning = ' '
        
        # if comments do not exist return index_df without changing
        if self.category is None:
            index_df.index = index_df.index.set_names(['file_id'])
            return index_df.reset_index(), com_warning
           
        # create empty arrays
        com_logic = np.zeros( (len(self.file_data), len(self.com_text_cols)*len(self.user_data)))
        com_time = np.zeros( (len(self.file_data), len(self.com_text_cols)*len(self.user_data)))
        
        # comment names
        com_names = [] 
        user_times = []
        
        cntr = 0 # init counter
        for i in range(len(self.user_data)): # iterate over user comment groups in category
            
            # get comment names, add number for repeated comments
            com_list = len(self.com_text_cols) * [self.user_data.at[i, 'Assigned Group Name']]
            com_names.extend(com_list)
            
            # get user selection
            user_time = self.user_data.at[i, 'Time Selection (sec)'].split(':')
            user_time = [int(x) for x in user_time]
            if len(user_time) != 2:
                raise Exception('Time could not be parsed for', self.user_data.at[i, 'Time Selection (sec)'])
            elif (user_time[1] - user_time[0]) < 1:
                raise Exception('Stop time must exceed start time.')
                
            # append to user times
            user_times.extend([np.array(user_time)] *len(self.com_text_cols))
            
            # get logic from all File comments
            temp_com_logic, temp_com_time = self.get_index_per_comment(i)
            
            # append to arrays
            com_logic[:,cntr:cntr + temp_com_logic.shape[1]] = temp_com_logic
            com_time[:,cntr:cntr + temp_com_logic.shape[1]] = temp_com_time   
            cntr += temp_com_logic.shape[1] 
        
        # check if at least one comment is present in each file
        if com_logic.any(axis=1).all() == False:
            com_warning = 'Comments were not detected in all files. Some data might be ommited from indexing.'
        
        # add present comments along with their time
        index_df = self.get_comments_with_time(index_df, com_names, user_times, com_logic.astype(bool), com_time)
        
        # if only one condition present remove last string
        last_string = index_df[self.category].str.strip().str[-1]
        if np.all(last_string == '1'):
            index_df[self.category] = index_df[self.category].str.strip().str[0:-1]
            
        # sort dataframe based on categories and file id
        index_df = index_df.rename_axis('file_id').sort_values(by=[self.category, 'file_id']).reset_index()
        
        return index_df, com_warning
        
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        