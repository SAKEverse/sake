# -*- coding: utf-8 -*-
import pandas as pd
from backend.filter_table import get_channel_structures, get_file_data, get_index_array
# from backend.adi_parse import AdiParse
# from backend.file_check import check_file
# from backend import search_function
# from backend.get_all_comments import GetComments
from backend.blocks_table import get_block_data, make_block_selector_panel

# define path
folder_path = r'R:\_protocols\_data_pipelines\1_LFP-EEG_Analysis\example_analysis'

# get user table data example
user_data = pd.read_csv(r"R:\_protocols\_data_pipelines\1_LFP-EEG_Analysis\example_analysis\user_data.csv")

# convert data frame to lower case
user_data = user_data.apply(lambda x: x.astype(str).str.lower())

# remove rows with no source
user_data = user_data[user_data.Source != '']

# get channel order
channel_structures = get_channel_structures(user_data)

# get block data
records = get_block_data(folder_path, channel_structures)

# select some blocks for testing (file_name: selected_block_index)
selected_blocks = dict(zip(['acuteallo#1-4-1.16.adicht', 'acuteallo#5-6-1.17.adicht',
 'acuteallobla_long1.adicht', 'acuteallobla_long2.adicht'] , [0, 2, 0, 0]))

# get all file data in dataframe
file_data = get_file_data(folder_path, channel_structures, selected_blocks=selected_blocks) #
print(file_data.block.unique())

# get experiment index
index_df, group_columns, warning_str = get_index_array(folder_path, user_data, file_data=file_data)
print(len(index_df))

