import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import dash_core_components as dcc
from app import app
from dash.dependencies import Input, Output


RF_table1 = pd.read_csv('../Result_csv/RF_table_res1.csv')
RF_table2 = pd.read_csv('../Result_csv/RF_table_res2.csv')
RF_table3 = pd.read_csv('../Result_csv/RF_table_res3.csv')
RF_table4 = pd.read_csv('../Result_csv/RF_table_res4.csv')
RF_table5 = pd.read_csv('../Result_csv/RF_table_res5.csv')
RF_table6 = pd.read_csv('../Result_csv/RF_table_res6.csv')
RF_table7 = pd.read_csv('../Result_csv/RF_table_res7.csv')
RF_table8 = pd.read_csv('../Result_csv/RF_table_res8.csv')
RF_table9 = pd.read_csv('../Result_csv/RF_table_res9.csv')
RF_table10 = pd.read_csv('../Result_csv/RF_table_res10.csv')

table1 = dbc.Table.from_dataframe(RF_table1,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table2 = dbc.Table.from_dataframe(RF_table2,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table3 = dbc.Table.from_dataframe(RF_table3,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table4 = dbc.Table.from_dataframe(RF_table4,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table5 = dbc.Table.from_dataframe(RF_table5,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table6 = dbc.Table.from_dataframe(RF_table6,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table7 = dbc.Table.from_dataframe(RF_table7,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table8 = dbc.Table.from_dataframe(RF_table8,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table9 = dbc.Table.from_dataframe(RF_table9,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table10 = dbc.Table.from_dataframe(RF_table10,bordered= True, dark= False, striped= True, hover = True,responsive= True )

layout=html.Div(children=[
    html.Br(),
    html.H4('The best forecasting model chosen according to the results of each model was found to be Random Forest'),
    html.Br(),
    
    html.H6('One of the goals of this work was to test if one model could be suitable for all house types and the results prove that it is indeed applicable'),
    html.H6('Several trials were done by changing some of the parameters of the Random Forest model in order to optimize the forecasted data'),

    html.H6('Trial 1: min_sample_leaf= 6  , n_estimators= 100'),
    html.Br(),
    html.H6('Trial 2: min_sample_leaf= 3  , n_estimators= 150, min_samples_split= 30'),
    html.H6('Trial 3: min_sample_leaf= 3  , n_estimators= 300, min_samples_split= 10, max_depth= 40'),

    html.Br(),
    html.Div(children=[
     dcc.RadioItems( id= 'radio1',
    options=[
        {'label': 'House 1', 'value': 'load1'},
        {'label': 'House 2', 'value': 'load2'},
        {'label': 'House 3', 'value': 'load3'},
        {'label': 'House 4', 'value': 'load4'},
        {'label': 'House 5', 'value': 'load5'},
        {'label': 'House 6', 'value': 'load6'},
        {'label': 'House 7', 'value': 'load7'},
        {'label': 'House 8', 'value': 'load8'},
        {'label': 'House 9', 'value': 'load9'},
        {'label': 'House 10', 'value': 'load10'},
    ],
    inputStyle={"margin-left": "20px"},
    value='load1',
    labelStyle={'display': 'inline-block'}
    )
    ]),
    html.Br(),
    html.Div(id= 'errortables1')
 ])
   
    
   

@app.callback(
    Output('errortables1', 'children'),
    [Input('radio1', 'value')]
    )
def errortable (value):
    if (value == 'load1'):
       return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table1),
            ])
    elif (value =='load2'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table2),
            ])
    elif (value =='load3'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table3),
            ])
    elif (value =='load4'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table4),
            ])
    elif (value =='load5'):
        return html.Div([
           html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table5),
            ])
    elif (value =='load6'):
        return html.Div([
           html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table6),
            ])
    elif (value =='load7'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table7),
            ])
    elif (value =='load8'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table8),
            ])
    elif (value =='load9'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table9),
            ])
    elif (value =='load10'):
        return html.Div([
           html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table10),
            ])