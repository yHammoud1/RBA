# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:09:05 2021

@author: Yara
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from app import app
import pandas as pd
import  plotly.express as px

df_raw = pd.read_csv('Proj3_clean_data_combined.csv')

df_raw_res = df_raw.drop(columns=['Date', 'Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])

df_raw_meteo = df_raw.drop(columns=['Date', 'Day of Week', 'Hour', 'Energy_res1', 'Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])

table_raw_res = df_raw.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])

table_raw_meteo = df_raw.drop(columns=['Energy_res1', 'Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])

table1 = dbc.Table.from_dataframe(table_raw_res[1:6],bordered= True, dark= False, hover = True,responsive= True, striped= True )

table2 = dbc.Table.from_dataframe(table_raw_meteo[1:6],bordered= True, dark= False, hover = True,responsive= True, striped= True )


layout= html.Div(children=[
    html.Br(),
    html.Br(),
    html.Br(),
    html.H4('Raw Energy Consumption Data of 10 Houses in Vancouver'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
         dcc.Tab(label='Raw Data Plots', value='tab-1', id='tab1', children =[
            html.Div(children = [
            html.Br(),
            html.H6('Raw data of energy consumption for all houses by house number'),
           
                dcc.Dropdown(
                    id="dropdown-1",
                    options=[{"label": i, "value": i} for i in df_raw_res.columns
                    ],
                    value="Energy_res1",
                ),
             ]),
            html.Div(children=[
          html.Div(
              dcc.Graph(id='raw-data-1'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
              dcc.Graph(id='boxplt-raw-1') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
                  ]
                    ),
             html.Br(),
            html.H6('Raw meteo data avialable in the region of study'),
          dcc.Dropdown(
                    id="dropdown-2",
                    options=[{"label": i, "value": i} for i in df_raw_meteo.columns
                    ],
                    value="AirTemp_C",
                ),
      
        html.Div(children=[
          html.Div(
        dcc.Graph(id='raw-data-2'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
          dcc.Graph(id='boxplt-raw-2') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
        ]
    ),
       ] 
    ),
            
         dcc.Tab(label='Raw Data Sample', value='tab-2', id='tab2', children = [
              html.Div(children = [
            html.Br(),
            html.H6('Sample of raw energy data and meteorological data'),
                dcc.Dropdown(id='dropdown-3',
                    options=[
                        {'label': 'Energy data', 'value': 'table1'},
                        {'label': 'Meteorological data', 'value': 'table2'}
                        ],
                    placeholder="Select a dataset to view a sample of the data",
                    ),
                html.Br(),
                html.Br(),
                html.Div(id='tables')

        ] 
    )
    
    
             ]),
     ]),
    
            ])
                



 

        

@app.callback(
    Output('raw-data-1', 'figure'),
    Output('boxplt-raw-1', 'figure'),
    [Input('dropdown-1', 'value')])
def prepare_raw_graphs(value):
     return {
        'data': [
            {'x': df_raw.Date, 'y': df_raw[value], 'type': 'scatter'},
        ],
        'layout': {
            'title': 'Raw data of ' + value 
        } }, px.box(df_raw, x=value)
 
@app.callback(
    Output('raw-data-2', 'figure'),
    Output('boxplt-raw-2', 'figure'),
    [Input('dropdown-2', 'value')])
def prepare_raw_graphs(value):
     return {
        'data': [
            {'x': df_raw.Date, 'y': df_raw[value], 'type': 'scatter'},
        ],
        'layout': {
            'title': 'Raw data of ' + value 
        } }, px.box(df_raw, x=value)


@app.callback(
    Output('tables', 'children'),
    [Input('dropdown-3', 'value')])
def print_tables(value):
      if (value == 'table1'):
        return html.Div(table1),
      elif (value == 'table2'):
          return html.Div(table2)