# -*- coding: utf-8 -*-

### --------------------------- Imports --------------------------- ###
import random
import pandas as pd
from backend.filter_table import get_channel_structures
from backend.blocks_table import get_block_data, make_block_selector_panel
### ----------------------------------------------------------------- ###

# define path
folder_path = r'R:\_protocols\_data_pipelines\1_LFP-EEG_Analysis\example_analysis'

# get user table data and channel structure
user_data = pd.read_csv(r"R:\_protocols\_data_pipelines\1_LFP-EEG_Analysis\example_analysis\user_data.csv")
user_data = user_data.apply(lambda x: x.astype(str).str.lower())
user_data = user_data[user_data.Source != '']
channel_structures = get_channel_structures(user_data)

# get block data
records = get_block_data(folder_path, channel_structures)

# create fake records for testing
def make_fake_records(n=20):
    records = []
    for i in range(n):
        n_blocks = random.randint(1, 4)
        blocks = [random.uniform(1000, 20000) for _ in range(n_blocks)]
        comments_by_block = {}
        for j, blen in enumerate(blocks):
            # put 0–3 random comments inside this block
            comments_by_block[j] = sorted(
                [random.uniform(0, blen) for _ in range(random.randint(0, 3))]
            )
        records.append({
            "file_name": f"fakefile_{i+1}.adicht",
            "blocks_s": blocks,
            "selected": random.randrange(n_blocks),
            "comments_by_block": comments_by_block,
        })
    return records

# Try with 20
fake_records = make_fake_records(20)

# Or preview the whole panel in a minimal Dash app
import dash
from dash import html
app = dash.Dash(__name__)
app.layout = html.Div([make_block_selector_panel(fake_records)])
app.run_server(debug=True, port=8051)
