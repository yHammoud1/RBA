import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from app import app
import pandas as pd
import  plotly.express as px
from sklearn.model_selection import train_test_split
from dash.dependencies import Input, Output

df_model_dataset1= pd.read_csv('../Result_csv/Model_dataset_res1.csv')
df_model_dataset2= pd.read_csv('../Result_csv/Model_dataset_res2.csv')
df_model_dataset3= pd.read_csv('../Result_csv/Model_dataset_res3.csv')
df_model_dataset4= pd.read_csv('../Result_csv/Model_dataset_res4.csv')
df_model_dataset5= pd.read_csv('../Result_csv/Model_dataset_res5.csv')
df_model_dataset6= pd.read_csv('../Result_csv/Model_dataset_res6.csv')
df_model_dataset7= pd.read_csv('../Result_csv/Model_dataset_res7.csv')
df_model_dataset8= pd.read_csv('../Result_csv/Model_dataset_res8.csv')
df_model_dataset9= pd.read_csv('../Result_csv/Model_dataset_res9.csv')
df_model_dataset10= pd.read_csv('../Result_csv/Model_dataset_res10.csv')

errors_table1= pd.read_csv('../Result_csv/errors_table_res1.csv')
errors_table2= pd.read_csv('../Result_csv/errors_table_res2.csv')
errors_table3= pd.read_csv('../Result_csv/errors_table_res3.csv')
errors_table4= pd.read_csv('../Result_csv/errors_table_res4.csv')
errors_table5= pd.read_csv('../Result_csv/errors_table_res5.csv')
errors_table6= pd.read_csv('../Result_csv/errors_table_res6.csv')
errors_table7= pd.read_csv('../Result_csv/errors_table_res7.csv')
errors_table8= pd.read_csv('../Result_csv/errors_table_res8.csv')
errors_table9= pd.read_csv('../Result_csv/errors_table_res9.csv')
errors_table10= pd.read_csv('../Result_csv/errors_table_res10.csv')

reg_table1= pd.read_csv('../Result_csv/reg_table_res1.csv')
reg_table2= pd.read_csv('../Result_csv/reg_table_res2.csv')
reg_table3= pd.read_csv('../Result_csv/reg_table_res3.csv')
reg_table4= pd.read_csv('../Result_csv/reg_table_res4.csv')
reg_table5= pd.read_csv('../Result_csv/reg_table_res5.csv')
reg_table6= pd.read_csv('../Result_csv/reg_table_res6.csv')
reg_table7= pd.read_csv('../Result_csv/reg_table_res7.csv')
reg_table8= pd.read_csv('../Result_csv/reg_table_res8.csv')
reg_table9= pd.read_csv('../Result_csv/reg_table_res9.csv')
reg_table10= pd.read_csv('../Result_csv/reg_table_res10.csv')


table_errors1 = dbc.Table.from_dataframe(errors_table1,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors2 = dbc.Table.from_dataframe(errors_table2,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors3 = dbc.Table.from_dataframe(errors_table3,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors4 = dbc.Table.from_dataframe(errors_table4,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors5 = dbc.Table.from_dataframe(errors_table5,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors6 = dbc.Table.from_dataframe(errors_table6,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors7 = dbc.Table.from_dataframe(errors_table7,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors8 = dbc.Table.from_dataframe(errors_table8,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors9 = dbc.Table.from_dataframe(errors_table9,bordered= True, dark= False, striped= True, hover = True,responsive= True )
table_errors10 = dbc.Table.from_dataframe(errors_table10,bordered= True, dark= False, striped= True, hover = True,responsive= True )

a_list = list(range(1, 200))

df_model_dataset1 = df_model_dataset1.drop(columns=['Date'])
X=df_model_dataset1.values
Y=X[:,0] #output is power
X=X[:,[1,2,3,4,5]]
X_train, X_test, y_train, y_test = train_test_split(X,Y)



layout=html.Div(children=[
     html.Br(), 
      html.H4('Testing Different Models to Forecast Energy Consumption'),
      html.Br(),
      dcc.Tabs(id='tabs', value='tab-1', children=[ 
         dcc.Tab(label='Models Predicted Data', value='tab-1', id='tab1', children =[ 
             html.Br(),
             html.H5('The plots below show the results of the different prediction models used for predicting the energy consumption of the houses.'),
             html.H6('The plots compare the predicted data (orange) with the testing data available (blue)'), 
             html.H6('Choose among the prediction models to view the results:'),
              dcc.Dropdown(
                    id="models",
                    options=[
                        {"label": i, "value": i } for i in errors_table1.Method 
                    ],
                    value="Random Forest", ),
             dcc.Graph(id='model-graphs'),
             ]),
         
         dcc.Tab(id= 'tab2', label='Model Errors', value='tab-2', children=[
             html.Br(),
             dcc.RadioItems( id= 'radio2',
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
    ),
             html.Div(id= 'errortables')
             ])
         ])
      ])


@app.callback(
     Output('model-graphs', 'figure'),
    [Input('models', 'value')])
def update_figure(value):
    if ( value == 'Linear Regression'):
        return  px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_LR[1:200]], color='y_pred_LR'),
           
    
    elif ( value == 'Random Forest'): 
        return  px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_RF[1:200]], color='variable')
    
    
    elif ( value == 'Support Vector Machine'):       
        return px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_SVR[1:200]], color='variable')

    elif ( value == 'Regression Decision Tree'):
        return  px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_DT[1:200]], color='variable')
    elif ( value == 'Random Forest Uniformized Data'):
        return  px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_RFU[1:200]], color='variable')
    elif ( value == 'Gradient Boosting'):
       return  px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_GB[1:200]], color='variable')
    elif ( value == 'Extreme Gradient Boosting'):
        return  px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_XGB[1:200]], color='variable')
    elif ( value == 'Bootsrapping'):
        return  px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_BT[1:200]], color='variable')
    elif ( value == 'Neural Networks'):
        return  px.line(reg_table1,x= a_list , y= [y_test[1:200],reg_table1.y_pred_NN[1:200]], color='variable')

       


@app.callback(
    Output('errortables', 'children'),
    [Input('radio2', 'value')]
    )
def errortable (value):
    if (value == 'load1'):
       return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors1),
            ])
    elif (value =='load2'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors2),
            ])
    elif (value =='load3'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors3),
            ])
    elif (value =='load4'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors4),
            ])
    elif (value =='load5'):
        return html.Div([
           html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors5),
            ])
    elif (value =='load6'):
        return html.Div([
           html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors6),
            ])
    elif (value =='load7'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors7),
            ])
    elif (value =='load8'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors8),
            ])
    elif (value =='load9'):
        return html.Div([
            html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors9),
            ])
    elif (value =='load10'):
        return html.Div([
           html.H5('The results of the errors produced by each model are presented in the below table'),
            html.Div(table_errors10),
            ])
