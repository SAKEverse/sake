# This is the main Dash app for SAKEPlan, integrating the block selection feature.
### --------------------------- Imports --------------------------- ###
import os
import json
import pandas as pd
import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from time import time as _now

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
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
app.server.secret_key = os.urandom(24)

# ---------------- Layout ------------------
app.layout = html.Div(
    [
        html.Div(children=main_layout),
        dcc.Store(id="blocksel_records"),
        dcc.Store(id="blocksel_store"),
        dcc.Store(id="refresh_token"),
        dcc.Store(id="user_df", storage_type="session"),
        dcc.Store(id="blocksel_cycle"),
    ]
)

# (Optional) validation layout
app.validation_layout = html.Div(
    [
        html.Div(children=main_layout),
        dcc.Store(id="blocksel_records"),
        dcc.Store(id="blocksel_store"),
        dcc.Store(id="refresh_token"),
        dcc.Store(id="user_df", storage_type="session"),
        dcc.Store(id="blocksel_cycle"),
    ]
)

# ========== User table session state ==========
@app.callback(
    [Output("user_df", "data"), Output("channel_name", "children")],
    [Input("user_table", "data")],
)
def update_user_data(table_data):
    df = pd.DataFrame(table_data)
    channels = df[df["Source"] == "channel_name"]
    categories = list(channels["Category"].unique())
    if "animal_id" in categories:
        categories.remove("animal_id")
    out_string = "Example Channel Name: -1001-"
    for category in categories:
        search_value = channels[channels["Category"] == category]["Search Value"]
        out_string += str(search_value.iloc[0])
    if "region" in df["Category"].values:
        regions = df[df["Category"] == "region"]["Assigned Group Name"]
        out_string += regions.iloc[0].split("-")[0]
    return df.to_json(date_format="iso", orient="split"), out_string

### ---------- Update User Table --------- ###
@app.callback(
    [
        Output("user_table", "columns"),
        Output("user_table", "data"),
        Output("user_table", "row_deletable"),
        Output("user_table", "dropdown"),
    ],
    [Input("add_row_button", "n_clicks"), Input("upload_data", "contents")],
    [State("user_df", "data")],
)
def update_usertable(n_clicks, upload_contents, session_user_data):
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "upload_data.contents":
        df = user_data_mod.upload_csv(upload_contents)
    else:
        if session_user_data is None:
            df = user_data_mod.original_user_data
        else:
            df = pd.read_json(session_user_data, orient="split")
    dash_cols, df, drop_dict = dashtable(df[user_data_mod.original_user_data.columns])
    if (n_clicks or 0) > 0:
        df = add_row(df)
    return dash_cols, df.to_dict("records"), True, drop_dict

# ---------- BLOCK SELECTOR: Accept button to store ---------- ###
# This store is used to emit a stable "Accept" signal that other callbacks can listen to
@app.callback(
    Output("blocksel_cycle", "data"),
    Input("blocksel_accept", "n_clicks"),
    prevent_initial_call=True,
)
def _accept_to_store(n):
    if not n:
        return no_update
    return _now()

### ---------- BLOCK SELECTOR: Build panel on Generate ---------- ###
# NOTE: we now listen to blocksel_cycle (stable store), not the disappearing button id
@app.callback(
    [Output("blocksel_panel_div", "children"), Output("blocksel_records", "data")],
    [Input("generate_button", "n_clicks"), Input("blocksel_cycle", "data")],
    [State("data_path_input", "value"), State("user_table", "data")],
)
def show_block_selector(n_clicks_generate, accept_token, folder_path, user_table_data):
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # ACCEPT signal via store: clear BOTH panel and records (so next Generate re-inits)
    if trig == "blocksel_cycle.data":
        return None, None

    # GENERATE path
    if not n_clicks_generate:
        return no_update, no_update
    if not folder_path or not user_table_data:
        return no_update, no_update

    df_user = pd.DataFrame(user_table_data)
    if "Source" not in df_user.columns:
        return no_update, no_update
    df_user = df_user.apply(lambda x: x.astype(str).str.lower())
    df_user = df_user[df_user["Source"] != ""]

    channel_structures = get_channel_structures(df_user)
    records = get_block_data(folder_path, channel_structures)
    if not records:
        return no_update, no_update

    panel = make_block_selector_panel(records, panel_id="blocksel")
    return panel, records


### ---------- BLOCK SELECTOR: Manage selection store ---------- ###
# Single owner: init on records, update on clicks, CLEAR on Accept (via store)
@app.callback(
    Output("blocksel_store", "data"),
    [
        Input("blocksel_records", "data"),
        Input({"type": "blocksel_graph", "file": ALL}, "clickData"),
        Input("blocksel_cycle", "data"),  # CHANGED: clear on Accept signal
    ],
    [State("blocksel_store", "data"), State({"type": "blocksel_graph", "file": ALL}, "id")],
)
def manage_blocksel_store(records, all_clicks, accept_token, current_store, all_ids):
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Clear selection on Accept signal
    if trig == "blocksel_cycle.data":
        return None

    # Initialize when records arrive
    if trig == "blocksel_records.data":
        if not records:
            return None
        return {r["file_name"]: int(r.get("selected", 0)) for r in records}

    # Update on any row click
    if trig.endswith(".clickData"):
        try:
            fired_json = trig.split(".")[0]
            fired_id = json.loads(fired_json)
            fname = fired_id.get("file")
        except Exception:
            return no_update

        clicked_block = None
        for i, cid in enumerate(all_ids or []):
            if isinstance(cid, dict) and cid.get("file") == fname:
                cd = (all_clicks or [None])[i]
                if cd and "points" in cd and cd["points"]:
                    point = cd["points"][0]
                    custom = point.get("customdata") or {}
                    clicked_block = (
                        int(custom["block"]) if isinstance(custom, dict) and "block" in custom
                        else int(point.get("curveNumber", 0))
                    )
                break

        if clicked_block is None or fname is None:
            return no_update

        new_store = dict(current_store or {})
        new_store[fname] = clicked_block
        return new_store

    return no_update


### ---------- BLOCK SELECTOR: Update a single row's figure ---------- ###
@app.callback(
    [Output({"type": "blocksel_graph", "file": MATCH}, "figure")],
    [Input({"type": "blocksel_graph", "file": MATCH}, "clickData")],
    [State("blocksel_records", "data"), State({"type": "blocksel_graph", "file": MATCH}, "id")],
)
def update_block_row_figure(click_data, records, this_id):
    if not click_data or not records or not this_id:
        return [no_update]
    file_name = this_id.get("file")
    point = click_data["points"][0]
    custom = point.get("customdata") or {}
    clicked_block = int(custom.get("block", point.get("curveNumber", 0)))
    by_file = {r["file_name"]: r for r in (records or [])}
    rec = by_file.get(file_name)
    if rec is None:
        return [no_update]
    fig = build_block_row_figure(rec, selected_override=clicked_block)
    return [fig]


### ---------- INDEX CREATION: Accept selection & build Sankey ---------- ###
# Accept → rebuild file_data using selected blocks, then produce Sankey & exports
# PLUS: emit an accept token into blocksel_cycle (so other callbacks can react safely)
@app.callback(
    [
        Output("alert_div", "children"),
        Output("tree_plot_div", "children"),
        Output("download_index_csv", "data"),
        Output("download_user_data_csv", "data"),
    ],
    [
        Input("blocksel_cycle", "data"),          # Accept signal (stable)
        Input("generate_button", "n_clicks"),     # Clear on Generate
    ],
    [
        State("data_path_input", "value"),
        State("user_table", "data"),
        State("blocksel_store", "data"),
    ],
)
def accept_build_or_clear(accept_token, n_generate, folder_path, user_data, selection):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update

    trig = ctx.triggered[0]["prop_id"]

    # Generate clicked: clear Sankey only
    if trig == "generate_button.n_clicks":
        return no_update, None, no_update, no_update

    # Accept signal: build Sankey & exports
    try:
        if not folder_path or not user_data:
            warn = dbc.Alert("Please provide a valid data path and user table.",
                             color="warning", dismissable=True)
            return warn, None, None, None

        index_df, group_names, warning_str = get_index_df(folder_path, user_data, selection)
        fig = dcc.Graph(id="tree_structure", figure=drawSankey(index_df[group_names]))

        data = dcc.send_data_frame(index_df.to_csv, "index.csv", index=False)
        user_data_df = pd.DataFrame(user_data)[user_data_mod.original_user_data.columns]
        user_data_export = dcc.send_data_frame(user_data_df.to_csv, "user_data.csv", index=False)

        warning = None if not (warning_str or "").strip() else dbc.Alert(
            id="alert_message", children=[str(warning_str)], color="warning", dismissable=True
        )
        return warning, fig, data, user_data_export

    except Exception as err:
        warning = dbc.Alert(id="alert_message", children=["   " + str(err)],
                            color="warning", dismissable=True)
        return warning, None, None, None


### ---------- BLOCK SELECTOR: Cancel (client-side page refresh) ---------- ###
app.clientside_callback(
    """
    function(n) {
        if (n) {
            window.location.reload();
            return Date.now();
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("refresh_token", "data"),
    Input("blocksel_cancel", "n_clicks"),
)

### ----------------- Run server ----------------- ###
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open("http://localhost:8050/", new=2)

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=False, port=8050, host="localhost")
### ----------------- End of File ----------------- ###
