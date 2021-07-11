# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 18:57:28 2021

@author: Yara
"""
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from app import app
import pandas as pd
import  plotly.express as px

df_zscore1 = pd.read_csv('EDA_zscore1.csv')
df_zscore2 = pd.read_csv('EDA_zscore2.csv')
df_zscore3 = pd.read_csv('EDA_zscore3.csv')


df_zscore1_opp = pd.read_csv('EDA_zscore1_opp.csv')
df_zscore2_opp = pd.read_csv('EDA_zscore2_opp.csv')
df_zscore3_opp = pd.read_csv('EDA_zscore3_opp.csv')



df_IQR4 = pd.read_csv('EDA_IQR4.csv')
df_IQR5 = pd.read_csv('EDA_IQR5.csv')
df_IQR6 = pd.read_csv('EDA_IQR6.csv')



df_IQR4_opp = pd.read_csv('EDA_IQR4_opp.csv')
df_IQR5_opp = pd.read_csv('EDA_IQR5_opp.csv')
df_IQR6_opp = pd.read_csv('EDA_IQR6_opp.csv')



df_EDA1_final = pd.read_csv('res11.csv')
df_EDA2_final = pd.read_csv('res21.csv')
df_EDA3_final = pd.read_csv('res31.csv')
df_EDA4_final = pd.read_csv('res41.csv')
df_EDA5_final = pd.read_csv('res51.csv')
df_EDA6_final = pd.read_csv('res61.csv')
df_EDA7_final = pd.read_csv('res71.csv')
df_EDA8_final = pd.read_csv('res81.csv')
df_EDA9_final = pd.read_csv('res91.csv')
df_EDA10_final = pd.read_csv('res101.csv')



layout= html.Div(children=[
    html.Br(),
    html.Br(),
    html.Br(),
    html.H4('Exploratory Data Analysis on Energy Data'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
         dcc.Tab(label='Z-score Method', value='tab-1', id='tab1', children =[
             html.Div(children = [
            html.Br(),
            html.H5('Removal of outliers by Z-score method'),
            html.H6('The results of a sample of the houses can be viewed'),
            
            dcc.Dropdown(
                    id="dropdown-1",
                    options=[{"label": 'House 1', "value": 'z1'},
                             {"label": 'House 2', "value": 'z2'},
                             {"label": 'House 3', "value": 'z3'},
                             
                             
                    ],
                    value="z1",
                ),
            
             html.Div(children=[
          html.Div(
        dcc.Graph(id='EDA-data1'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
          dcc.Graph(id='boxplt-EDA1') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          ]),
            
            ])
             ]),
         
         dcc.Tab(label='IQR Method', value='tab-2', id='tab2', children = [
              html.Div(children = [
            html.Br(),
            html.H6('Removal of outliers by IQR method'),
            html.H6('The results of a sample of the houses can be viewed'),
            
            dcc.Dropdown(
                    id="dropdown-2",
                    options=[
                             {"label": 'House 4', "value": 'i4'},
                             {"label": 'House 5', "value": 'i5'},
                             {"label": 'House 6', "value": 'i6'},
                             
                             
                    ],
                    value="i4",
                ),
            
             html.Div(children=[
          html.Div(
        dcc.Graph(id='EDA-data2'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
          dcc.Graph(id='boxplt-EDA2') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          ]),
            
            ])
              ]),
         
         dcc.Tab(label='Final Outliers Removal', value='tab-3', id='tab3', children = [
              html.Div(children = [
            html.Br(),
            html.H5('Final Decision of outliers to be removed'),
            html.H6('The plots show the data after the removal'),
            html.H6('The results of a sample of the houses can be viewed'),
            
            
            dcc.Dropdown(
                    id="dropdown-3",
                    options=[
                             {"label": 'House 7', "value": 'f7'},
                             {"label": 'House 8', "value": 'f8'},
                             {"label": 'House 9', "value": 'f9'},
                           
                             
                    ],
                    value="f7",
                ),
            
             html.Div(children=[
          html.Div(
        dcc.Graph(id='EDA-data3'), style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          html.Div(
          dcc.Graph(id='boxplt-EDA3') , style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}
              ),
          ]),
            
            ])
              ]),
            
    ]), 
    
])

@app.callback(
    Output('EDA-data1', 'figure'),
    Output('boxplt-EDA1', 'figure'),
    [Input('dropdown-1', 'value')])
def prepare_eda_graphs1(value):
    if (value == 'z1'): 
        return {
        'data': [
            {'x': df_zscore1.Date, 'y': df_zscore1['Energy_res1'], 'type': 'scatter', 'name' : 'Correct Data'}, 
            {'x': df_zscore1_opp.Date, 'y': df_zscore1_opp['Energy_res1'], 'type': 'scatter', 'name' : 'Outliers'},
        ],
        'layout': {
            'title': 'Outliers found by Z-score method '  
        } }, px.box(df_zscore1, x=df_zscore1['Energy_res1'], title = 'Boxplot after outliers removal')
    elif (value == 'z2'):
        return {
        'data': [
            {'x': df_zscore2.Date, 'y': df_zscore2['Energy_res2'], 'type': 'scatter', 'name' : 'Correct Data'}, 
            {'x': df_zscore2_opp.Date, 'y': df_zscore2_opp['Energy_res2'], 'type': 'scatter', 'name' : 'Outliers'},
        ],
        'layout': {
            'title': 'Outliers found by Z-score method '  
        } }, px.box(df_zscore2, x=df_zscore2['Energy_res2'], title = 'Boxplot after outliers removal')
    elif (value == 'z3'):
        return {
        'data': [
            {'x': df_zscore3.Date, 'y': df_zscore3['Energy_res3'], 'type': 'scatter', 'name' : 'Correct Data'}, 
            {'x': df_zscore3_opp.Date, 'y': df_zscore3_opp['Energy_res3'], 'type': 'scatter', 'name' : 'Outliers'},
        ],
        'layout': {
            'title': 'Outliers found Z-score method '  
        } }, px.box(df_zscore3, x=df_zscore3['Energy_res3'], title = 'Boxplot after outliers removal')
    
 
         
@app.callback(
    Output('EDA-data2', 'figure'),
    Output('boxplt-EDA2', 'figure'),
    [Input('dropdown-2', 'value')])
def prepare_eda_graphs2(value):
    if (value == 'i4'): 
        return {
        'data': [
            {'x': df_IQR4.Date, 'y': df_IQR4['Energy_res4'], 'type': 'scatter', 'name' : 'Correct Data'},
            {'x': df_IQR4_opp.Date, 'y': df_IQR4_opp['Energy_res4'], 'type': 'scatter', 'name' : 'Outliers'},
        ],
        'layout': {
            'title': 'Outliers found IQR method '  
        } }, px.box(df_IQR4, x=df_IQR4['Energy_res4'], title = 'Boxplot after outliers removal')
    elif (value == 'i5'):
        return {
        'data': [
            {'x': df_IQR5.Date, 'y': df_IQR5['Energy_res5'], 'type': 'scatter', 'name' : 'Correct Data'}, 
            {'x': df_IQR5_opp.Date, 'y': df_IQR5_opp['Energy_res5'], 'type': 'scatter', 'name' : 'Outliers'},
        ],
        'layout': {
            'title': 'Outliers found by IQR method '  
        } }, px.box(df_IQR5, x=df_IQR5['Energy_res5'], title = 'Boxplot after outliers removal')
    elif (value == 'i6'):
        return {
        'data': [
            {'x': df_IQR6.Date, 'y': df_IQR6['Energy_res6'], 'type': 'scatter', 'name' : 'Correct Data'},
            {'x': df_IQR6_opp.Date, 'y': df_IQR6_opp['Energy_res6'], 'type': 'scatter', 'name' : 'Outliers'},
        ],
        'layout': {
            'title': 'Outliers found by IQR method '  
        } }, px.box(df_IQR6, x=df_IQR6['Energy_res6'], title = 'Boxplot after outliers removal')
    
    
@app.callback(
    Output('EDA-data3', 'figure'),
    Output('boxplt-EDA3', 'figure'),
    [Input('dropdown-3', 'value')])
def prepare_eda_graphs3(value):
    if (value == 'f7'): 
        return {
        'data': [
            {'x': df_EDA7_final.Date, 'y': df_EDA7_final['Energy_res7'], 'type': 'scatter'},      
        ],
        'layout': {
            'title': 'Final Outliers removal '  
        } }, px.box(df_EDA7_final, x=df_EDA7_final['Energy_res7'], title = 'Boxplot after outliers removal')
    elif (value == 'f8'):
        return {
        'data': [
            {'x': df_EDA8_final.Date, 'y': df_EDA8_final['Energy_res8'], 'type': 'scatter'},      
        ],
        'layout': {
            'title': 'Final Outliers removal '  
        } }, px.box(df_EDA8_final, x=df_EDA8_final['Energy_res8'], title = 'Boxplot after outliers removal')
    elif (value == 'f9'):
        return {
        'data': [
            {'x': df_EDA9_final.Date, 'y': df_EDA9_final['Energy_res9'], 'type': 'scatter'},      
        ],
        'layout': {
            'title': 'Final Outliers removal '  
        } }, px.box(df_EDA9_final, x=df_EDA9_final['Energy_res9'], title = 'Boxplot after outliers removal')
