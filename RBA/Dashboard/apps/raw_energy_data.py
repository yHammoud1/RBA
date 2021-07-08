# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:09:05 2021

@author: Yara
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from app import app
import pandas as pd
import  plotly.express as px

df_raw = pd.read_csv('Proj3_clean_data_combined.csv')

df_raw_res = df_raw.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])

df_raw_meteo = df_raw.drop(columns=['Energy_res1', 'Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])



layout= html.Div(children=[
    html.Br(),
    html.Br(),
    html.Br(),
    dcc.Tabs(id='tabs', value='tab-1', children=[
         dcc.Tab(label='Raw Data Plots', value='tab-1'),
         dcc.Tab(label='Raw Data Sample', value='tab-2'),
     ]),
    html.Div(id = 'content')
            ])
                


#             dbc.FormGroup(
#             [
#                 dcc.Dropdown(
#                     id="dropdown-1",
#                     options=[{"label": i, "value": i} for i in df_raw_res.columns
#                     ],
#                     value="Energy_res1",
#                 ),
#             ]
#         ),

#          html.Div(children=[
#           html.Div(
#              dcc.Graph(id='raw-data'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
#               ),
#           html.Div(
#              dcc.Graph(id='boxplt-raw') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
#               ),
#         ]
#     ),
# ]),
    
    
#             dbc.FormGroup(
#             [
#                 dcc.Dropdown(
#                     id="dropdown-1",
#                     options=[{"label": i, "value": i} for i in df_raw_meteo.columns
#                     ],
#                     value="AirTemp_C",
#                 ),
#             ]
#         ),

#       html.Div(children=[
#           html.Div(
#         dcc.Graph(id='raw-data'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
#               ),
#           html.Div(
#           dcc.Graph(id='boxplt-raw') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
#               ),
#         ]
#     ),
        
#     ]
# )

@app.callback(
    Output('content', 'children'),
    [Input('tabs', 'value')])
def show_content(value):
    if (value == 'tab-1'):
        return html.Div(children = [
           
                dcc.Dropdown(
                    id="dropdown-1",
                    options=[{"label": i, "value": i} for i in df_raw_res.columns
                    ],
                    value="Energy_res1",
                ),
            
            

          html.Div(children=[
          html.Div(
              dcc.Graph(id='raw-data'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
              dcc.Graph(id='boxplt-raw') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
                  ]
                    ),
                  
          dcc.Dropdown(
                    id="dropdown-1",
                    options=[{"label": i, "value": i} for i in df_raw_meteo.columns
                    ],
                    value="AirTemp_C",
                ),
      
    html.Div(children=[
          html.Div(
        dcc.Graph(id='raw-data'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
          dcc.Graph(id='boxplt-raw') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
        ]
    ),
       ] 
    )

    
            
    elif (value == 'tab-2'):
        return  html.Div(children = [
           
                dcc.Dropdown(
                    id="dropdown-1",
                    options=[{"label": i, "value": i} for i in df_raw_res.columns
                    ],
                    value="Energy_res1",
                ),
            
            

          html.Div(children=[
          html.Div(
              dcc.Graph(id='raw-data'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
              dcc.Graph(id='boxplt-raw') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
                  ]
                    ),
                  
          dcc.Dropdown(
                    id="dropdown-1",
                    options=[{"label": i, "value": i} for i in df_raw_meteo.columns
                    ],
                    value="AirTemp_C",
                ),
      
    html.Div(children=[
          html.Div(
        dcc.Graph(id='raw-data'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
          dcc.Graph(id='boxplt-raw') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
        ]
    ),
       ] 
    )
    

    
    
    

@app.callback(
    Output('raw-data', 'figure'),
    Output('boxplt-raw', 'figure'),
    [Input('dropdown-1', 'value')])
def prepare_raw_graphs(value):
     return {
        'data': [
            {'x': df_raw.Date, 'y': df_raw[value], 'type': 'scatter'},
        ],
        'layout': {
            'title': 'Raw data of ' + value 
        } }, px.box(df_raw, x=value)
