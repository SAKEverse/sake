# This is the main Dash app for SAKEPlan, integrating the block selection feature.
### --------------------------- Imports --------------------------- ###
import os
import pandas as pd
import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, MATCH
import dash_bootstrap_components as dbc
# ---- SAKE modules ----
from layouts import main_layout
from backend.create_user_table import dashtable, add_row
from backend.tree import drawSankey
from backend.filter_table import (
    get_index_df,
    get_channel_structures,
)
from backend.blocks_table import (
    get_block_data,
    make_block_selector_panel,
    build_block_row_figure,
)
import user_data_mod
### ----------------------------------------------------------------- ###

# ---------------- App boot ----------------
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True,)
app.server.secret_key = os.urandom(24)

# ---------------- Layout ------------------
# We  layout1 (previous UI), then add:
#   - blocksel_records: holds the records (list[dict]) for the selector
#   - blocksel_store:   holds {file_name: selected_block}
app.layout = html.Div(
    [html.Div(children=main_layout),
    dcc.Store(id="blocksel_records"),
    dcc.Store(id="blocksel_store"),
    dcc.Store(id="refresh_token"),  # dummy to trigger clientside callback
    dcc.Store(id='user_df', storage_type='session'),
    ]
)

# update user data in session
@app.callback(
    [Output('user_df', 'data'), Output('channel_name', 'children')],
    [Input('user_table', 'data'),]
    )
def update_user_data(table_data):
    
    # get channel strings
    df = pd.DataFrame(table_data)
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
    [Output('user_table', "columns"), Output('user_table', 'data'),
    Output('user_table', 'row_deletable'), Output('user_table', 'dropdown')],
    [Input("add_row_button","n_clicks"),Input('upload_data', 'contents'),],
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
            df = user_data_mod.original_user_data # get default dataframe
        else:                           # load user input from current session
            df = pd.read_json(session_user_data, orient='split') # get data from user datatabl

    # convert user data in dashtable format
    dash_cols, df, drop_dict = dashtable(df[user_data_mod.original_user_data.columns]) 

    if n_clicks > 0: # Add rows when button is clicked
        df = add_row(df)

    return dash_cols, df.to_dict('records'), True, drop_dict
### ----------------------------------- ###

# ========== BLOCK SELECTOR: Build panel on Generate ==========
@app.callback(
    [Output("blocksel_panel_div", "children"),
     Output("blocksel_records", "data"),
     Output("blocksel_store", "data")],
    [Input("generate_button", "n_clicks"),
     Input("blocksel_accept", "n_clicks")],
    [State("data_path_input", "value"),
     State("user_table", "data")],
)
def show_block_selector(n_clicks_generate, n_clicks_accept, folder_path, user_table_data):
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # If ACCEPT was clicked, clear the panel and keep stores as-is
    if trig == "blocksel_accept.n_clicks":
        return None, no_update, no_update

    # GENERATE path below is unchanged
    if not n_clicks_generate:
        return no_update, no_update, no_update
    if not folder_path or not user_table_data:
        return no_update, no_update, no_update

    df_user = pd.DataFrame(user_table_data)
    if "Source" not in df_user.columns:
        return no_update, no_update, no_update
    df_user = df_user.apply(lambda x: x.astype(str).str.lower())
    df_user = df_user[df_user["Source"] != ""]

    channel_structures = get_channel_structures(df_user)
    records = get_block_data(folder_path, channel_structures)
    if not records:
        return no_update, no_update, no_update

    selection = {r["file_name"]: int(r.get("selected", 0)) for r in records}
    panel = make_block_selector_panel(records, panel_id="blocksel")

    return panel, records, selection


# ========== BLOCK SELECTOR: Update a single row on click ==========
@app.callback(
    [Output({"type": "blocksel_graph", "file": MATCH}, "figure")],
    [Input({"type": "blocksel_graph", "file": MATCH}, "clickData")],
    [State("blocksel_records", "data"),
     State("blocksel_store", "data"),
     State({"type": "blocksel_graph", "file": MATCH}, "id")],
)
def update_block_row_figure(click_data, records, selection, this_id):
    # no click / no state -> no change
    if not click_data or not records or not this_id:
        return [no_update]

    file_name = this_id.get("file")

    # get clicked block index from customdata
    point = click_data["points"][0]
    custom = point.get("customdata") or {}
    clicked_block = int(custom["block"]) if isinstance(custom, dict) and "block" in custom \
                    else int(point.get("curveNumber", 0))

    # fetch the record for this file and rebuild only this row
    by_file = {r["file_name"]: r for r in (records or [])}
    rec = by_file.get(file_name)
    if rec is None:
        return [no_update]

    fig = build_block_row_figure(rec, selected_override=clicked_block)
    return [fig]


# ========== INDEX CREATION: Accept selection & build Sankey ==========
# Accept → rebuild file_data using selected blocks, then produce Sankey & exports
@app.callback(
    [Output('alert_div', 'children'),
     Output('tree_plot_div', 'children'),
     Output('download_index_csv', 'data'),
     Output('download_user_data_csv', 'data'),
     ],
    [Input('blocksel_accept', 'n_clicks')],
    [State('data_path_input', 'value'),
     State('user_table', 'data'),
     State('blocksel_store', 'data')],
)
def accept_and_build(n_clicks, folder_path, user_data, selection):
    if not n_clicks:
        # no-op until Accept is clicked
        return no_update, no_update, no_update, no_update

    try:
        # Guard: need a path and user table
        if not folder_path or not user_data:
            warn = dbc.Alert("Please provide a valid data path and user table.", color="warning", dismissable=True)
            return warn, None, None, None

        # get index df
        index_df, group_names, warning_str = get_index_df(folder_path, user_data, selection)

        # Sankey plot
        fig = dcc.Graph(id='tree_structure', figure=drawSankey(index_df[group_names]))

        # Exports
        data = dcc.send_data_frame(index_df.to_csv, 'index.csv', index=False)
        user_data = pd.DataFrame(user_data)
        user_data = user_data[user_data_mod.original_user_data.columns] 
        user_data_export = dcc.send_data_frame(user_data.to_csv, 'user_data.csv', index = False)

        # Warning banner (if any)
        warning = None if not (warning_str or "").strip() \
                 else dbc.Alert(id='alert_message', children=[str(warning_str)], color="warning", dismissable=True)

        # Hide the selector panel after accept
        return warning, fig, data, user_data_export

    except Exception as err:
        warning = dbc.Alert(id='alert_message', children=['   ' + str(err)], color="warning", dismissable=True)
        return warning, None, None, None

# # ========== BLOCK SELECTOR: Cancel ==========
app.clientside_callback(
    """
    function(n) {
        if (n) {
            window.location.reload();
            // return something so Dash is happy
            return Date.now();
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("refresh_token", "data"),
    Input("blocksel_cancel", "n_clicks"),
)


### ----------------- Run server ----------------- ###

# Automatic browser launch
import webbrowser
from threading import Timer
def open_browser():
      webbrowser.open('http://localhost:8050/', new = 2)

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=8050, host='localhost')








