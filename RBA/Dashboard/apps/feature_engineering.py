# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 22:39:56 2021

@author: Yara
"""
import dash_html_components as html
import dash_bootstrap_components as dbc

import pandas as pd
import dash_core_components as dcc

df_feature_eng= pd.read_csv('../Result_csv/Feature_Eng_res1.csv')

df_model_dataset= pd.read_csv('../Result_csv/Model_dataset_res1.csv')

def fix_tables(df):
    dff = df.Date
    df = df.drop(columns=['Date'])
    df = df.round(2)
    df.insert(0, 'Date', dff)
    return df

table2 = fix_tables(df_feature_eng)
table3 = fix_tables(df_model_dataset)

table = dbc.Table.from_dataframe(table2[1:6],bordered= True, dark= False, striped= True, hover = True,responsive= True )
table1 = dbc.Table.from_dataframe(table3[1:6],bordered= True, dark= False, striped= True, hover = True,responsive= True )

layout=html.Div(children=[
     html.Br(), 
      html.Br(), 
      html.Br(),
      html.H4('Feature Engineering/Extraction'),
      html.H5('House 1 data are presented as an example'),
      dcc.Tabs(id='tabs', value='tab-1', children=[ 
         dcc.Tab(label='Engineered Features', value='tab-1', id='tab1', children =[ 
             html.H5('New features were engineered to optimize the modeling using the available data: Temp2, LWE2, Heating degree.hour, Prec2 and Energy-1'),
             html.H6('explain the equations used ----------------'),
             html.Div(table),
             ]),
         
         dcc.Tab(label='Model Dataset', value='tab-2', children=[
              html.H5('Final features selected to perform the modelling are presented in the following table'),
              html.H6('After applying Filter, Wrapper and Ensemble methods, the most relevant features selected were: Energy-1, hour, temperature, Solar radiation, Day of Week'),
              html.Div(table1),
             ])
         ])
])
        

