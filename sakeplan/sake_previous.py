### ---------------------------- Imports ---------------------------- ###
import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# User Defined #
from app import app
from layouts import layout1
import dash_bootstrap_components as dbc
from backend.create_user_table import dashtable, add_row
from backend.tree import drawSankey
from backend.filter_table import get_index_array
import user_data_mod
### ----------------------------------------------------------------- ###

# init dash app with css style sheets
# apply general css styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
app.server.secret_key = os.urandom(24)

# Define main layout
app.layout = html.Div(children = [
    
    html.Div(children = layout1), 
])

# update user data in session
@app.callback(
    [Output('user_df', 'data'),
    Output('channel_name', 'children')],
    [Input('user_table', 'data'),]
   
    )
def update_user_data(table_data):
    
    df = pd.DataFrame(table_data)

    # get channel strings
    channels = df[df['Source'] =='channel_name']
    categories = list(channels['Category'].unique())
    if 'animal_id' in categories: 
        categories.remove('animal_id')
    out_string='Example Channel Name: -1001-'
    for category in categories: 
        search_value = channels[channels['Category']==category]['Search Value']
        out_string+= str(search_value.iloc[0])

    # add brain region
    if 'region' in df['Category'].values:
        regions = df[df['Category'] =='region']['Assigned Group Name']
        out_string += regions.iloc[0].split('-')[0]

    return df.to_json(date_format='iso', orient='split'), out_string


### ---------- Update User Table--------- ###
@app.callback(
    [Output('user_table', "columns"), 
    Output('user_table', 'data'),
    Output('user_table', 'row_deletable'),
    Output('user_table', 'dropdown')
    ],
    [Input("add_row_button","n_clicks"),
    Input('upload_data', 'contents'),],
    [State('user_df', 'data')],
    
    )
def update_usertable(n_clicks, upload_contents, session_user_data):

    # get context
    ctx = dash.callback_context
    
    # load user input from csv file selected by user
    if 'upload_data.contents' in ctx.triggered[0]['prop_id']:

        df = user_data_mod.upload_csv(upload_contents)
    else:
        if session_user_data == None:   # if new user session
            # get default dataframe
            df = user_data_mod.original_user_data
        else:                           # load user input from current session
            # get data from user datatable
            df = pd.read_json(session_user_data, orient='split')

    # convert user data in dashtable format
    dash_cols, df, drop_dict = dashtable(df[user_data_mod.original_user_data.columns]) 

    if n_clicks > 0: # Add rows when button is clicked
        df = add_row(df)

    return dash_cols, df.to_dict('records'), True, drop_dict


# Retrieve path and plot tree diagram
@app.callback(
    [Output('alert_div', 'children'),
     Output('tree_plot_div', 'children'),
     Output('download_index_csv', 'data'),
     Output('download_user_data_csv', 'data')],
    [Input('generate_button', 'n_clicks')],
    [State('data_path_input', 'value'),
    State('user_table', 'data')],
)
def update_output(n_clicks1, folder_path, user_data):

    try:
        if folder_path is None:         
            warning = None
            fig = None
            data = None
            user_data_export = None
        else:

            # get grouped dataframe
            index_df, group_names, warning_str = get_index_array(folder_path, user_data)

            # Get tree plot as dcc graph
            fig = dcc.Graph(id = 'tree_structure', figure = drawSankey(index_df[group_names]))

            # send index_df for download
            data = dcc.send_data_frame(index_df.to_csv, 'index.csv', index = False)

            # send user data for download
            user_data = pd.DataFrame(user_data)
            user_data = user_data[user_data_mod.original_user_data.columns] 
            user_data_export = dcc.send_data_frame(user_data.to_csv, 'user_data.csv', index = False)

            # if warning_str set to none so that no warning is shown in sake app
            if len(warning_str.strip()) == 0:
                warning = None
            else:
                warning = dbc.Alert(id = 'alert_message', children = [str(warning_str)], color="warning", dismissable=True)

        return warning, fig, data, user_data_export

    except Exception as err:
        warning = dbc.Alert(id = 'alert_message', children = ['   ' + str(err)], color="warning", dismissable=True) #, duration = 10000
        return warning, None, None, None

# Automatic browser launch
import webbrowser
from threading import Timer
def open_browser():
      webbrowser.open('http://localhost:8050/', new = 2)

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug = False,
                    port = 8050,
                    host = 'localhost',
                   )








