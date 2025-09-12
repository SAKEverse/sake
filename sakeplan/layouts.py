
### --------------- IMPORTS --------------- ###
from dash import dcc, html, dash_table
### --------------------------------------- ###

# Always present
main_layout =  html.Div(id = 'layout_channel', children=[

        
    html.Div([ # MAIN BANNER
        html.Img(src='/assets/sakeico.png', id = 'banner_img'),
        html.Div(children = ['SAKEplan'], id = 'banner_text'),
        
    ], className='main_banner'),

    # 0- alerts
    html.Div(id='alert_div', children =[ # show warnings
        
    ]),

    # 1- generate button + folder path field + upload settings
    html.Div(id='generate_plus_field_div', children=[

        html.Div(id='export_div', children=[ # invisible (used for export)
            dcc.Download(id='download_index_csv') ]),

        html.Div(id='export_user_data_div', children=[ # invisible (used for export)
            dcc.Download(id='download_user_data_csv') ]),

        html.Div( id='generate_div', children=[
            html.Button('Generate', id='generate_button', n_clicks=0,   
            ),]),

        html.Div(id='data_path_main_div', children = [ 
            dcc.Input(id='data_path_input', type='text', placeholder='Path to data folder'),
        ]),

        html.Div(id='load_user_data_div', children=[
            dcc.Upload( id='upload_data', accept = '.csv', children = (html.Button('load_settings', id='load_settings', n_clicks=0))),
        ]),

    ]),

    # generate example channel name on top of table
    html.Div(id='channel_name'),

    html.Div(id = 'drop_message', children=[ "*** To drop channels rename 'Assigned Group Name' to 'drop'"]
    ),
    
    # 2 create user table
    html.Div(id='user_table_div', children=[

        dash_table.DataTable(id = 'user_table',
                    editable = True,
                    style_cell={
                            'color': 'black',
                            'textAlign': 'center',
                            'font-family':'arial',
                            'font-size': '100%',
                            },
                    style_header={
                        'fontWeight': 'bold',
                        'color': 'black',
                        'textAlign': 'center',
                        'backgroundColor': 'rgb(230, 230, 230)',
                            },
                    style_data={
                        'width': '150', 'minWidth': '100', 'maxWidth': '175'}
            )
    ]),

    # 2- add row button
    html.Div( id='add_row_button_div', children=[
        html.Button('+', id='add_row_button', n_clicks=0,   
        ),]),

    # 4- tree plot
    html.Div(id='tree_plot_div', children=[]),

    # 3- block selection plot
    html.Div(id="blocksel_panel_div", children=[]),
    html.Button(id="blocksel_accept", style={"display": "none"}),
    html.Button(id="blocksel_cancel", style={"display": "none"}),

])###