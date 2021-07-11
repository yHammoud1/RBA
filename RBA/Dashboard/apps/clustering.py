import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from app import app

import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from dash.dependencies import Input, Output

df_kmeans1_res1= pd.read_csv('../Result_csv/Cluster_kmeans1_res1.csv')
df_kmeans1_res3= pd.read_csv('../Result_csv/Cluster_kmeans1_res3.csv')
df_kmeans1_res5= pd.read_csv('../Result_csv/Cluster_kmeans1_res5.csv')
df_kmeans1_res7= pd.read_csv('../Result_csv/Cluster_kmeans1_res7.csv')
df_kmeans1_res9= pd.read_csv('../Result_csv/Cluster_kmeans1_res9.csv')
df_kmeans1_res10= pd.read_csv('../Result_csv/Cluster_kmeans1_res10.csv')

df_kmeans2_res1= pd.read_csv('../Result_csv/Cluster_kmeans2_res1.csv')
df_kmeans2_res3= pd.read_csv('../Result_csv/Cluster_kmeans2_res3.csv')
df_kmeans2_res5= pd.read_csv('../Result_csv/Cluster_kmeans2_res5.csv')
df_kmeans2_res7= pd.read_csv('../Result_csv/Cluster_kmeans2_res7.csv')
df_kmeans2_res9= pd.read_csv('../Result_csv/Cluster_kmeans2_res9.csv')
df_kmeans2_res10= pd.read_csv('../Result_csv/Cluster_kmeans2_res10.csv')

df_kmeans1_res1 = df_kmeans1_res1.rename(columns={'Energy_res1':'Energy_res'})
df_kmeans1_res3 = df_kmeans1_res3.rename(columns={'Energy_res3':'Energy_res'})
df_kmeans1_res5 = df_kmeans1_res5.rename(columns={'Energy_res5':'Energy_res'})
df_kmeans1_res7 = df_kmeans1_res7.rename(columns={'Energy_res7':'Energy_res'})
df_kmeans1_res9 = df_kmeans1_res9.rename(columns={'Energy_res9':'Energy_res'})
df_kmeans1_res10 = df_kmeans1_res10.rename(columns={'Energy_res10':'Energy_res'})

df_kmeans2_res1 = df_kmeans2_res1.rename(columns={'Energy_res1':'Energy_res'})
df_kmeans2_res3 = df_kmeans2_res3.rename(columns={'Energy_res3':'Energy_res'})
df_kmeans2_res5 = df_kmeans2_res5.rename(columns={'Energy_res5':'Energy_res'})
df_kmeans2_res7 = df_kmeans2_res7.rename(columns={'Energy_res7':'Energy_res'})
df_kmeans2_res9 = df_kmeans2_res9.rename(columns={'Energy_res9':'Energy_res'})
df_kmeans2_res10 = df_kmeans2_res10.rename(columns={'Energy_res10':'Energy_res'})

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x-variable",
                    options=[
                        {"label": 'Energy', "value":'Energy_res' }
                         
                    ],
                    value="Energy_res",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="y-variable",
                    options=[
                        {"label": 'Temperature', "value":'AirTemp_C' },
                        {"label": 'Hour', "value":'Hour' }, 
                        {"label": 'Week Day', "value":'Day of Week' }
                        
                    ],
                    value="AirTemp_C",
                ),
            ]
        ),
        
    ],
    body=True,
)

controls1 = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x-variable1",
                    options=[
                        {"label": 'Energy', "value":'Energy_res' },
                    ],
                    value="Energy_res",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="y-variable1",
                    options=[
                        {"label": 'Relative Humidity', "value": 'RelativeHumidity' },
                        {"label": 'Wind Speed', "value":'WindSpeed10m_m/s' },
                        {"label": 'Solar Radiation', "value":'GlobalSolarRad_W/m2' },
                    ],
                    value="RelativeHumidity",
                ),
            ]
        ),
        
    ],
    body=True,
) 


layout=html.Div(children=[
    html.Br(), 
    html.H4('Clustering of Data Using Kmeans and Clustered Load Curve Results'),
    html.Br(),
    dcc.Tabs(id='tabs-1', value='tab-1', children=[
         dcc.Tab(label='Kmeans', value='tab-1', id='kmeanstab', children=[
             html.Div([
                 html.Br(),
       html.H5('Clustering using Kmeans: trials with sets of different parameters'),
        html.Br(), 
       
            dcc.RadioItems( id= 'radio1',
    options=[
        {'label': 'House 1', 'value': 'house1'},
        {'label': 'House 3', 'value': 'house3'},
        {'label': 'House 5', 'value': 'house5'},
        {'label': 'House 7', 'value': 'house7'},
        {'label': 'House 9', 'value': 'house9'},
        {'label': 'House 10', 'value': 'house10'},
    ],
    inputStyle={"margin-left": "20px"},
    value='house1',
    labelStyle={'display': 'inline-block'}
    ), 
            
        html.H5('Trial 1: Energy, Temperature, Weekday and Hour'),
        html.H6('Optimal number of clusters was found to be 9'),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(dcc.Graph(id="cluster-graph"), md=8),
            ],
            align="center",
                ),
        
        html.H5('Trial 2: Energy, Humidity, Wind speed and Solar radiation'),
        html.H6('Optimal number of clusters was found to be 3'),
        dbc.Row(
            [
                dbc.Col(controls1, md=4),
                dbc.Col(dcc.Graph(id="cluster-graph2"), md=8),
            ],
            align="center",
                ),
        ])
             ]),
         
         
         dcc.Tab(label='Load Curve', value='tab-2', children=[
             
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
             html.Br(),
             
             html.Div(id= 'loadcurves')
             ]),
     ]),
    
    html.Div(id='tabs-content'), 
    ])

@app.callback(
    Output('cluster-graph', 'figure'),
    [Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('radio1', 'value')])
def make_graph(x, y, z):
    if (z == 'house1'):
        km = KMeans(max(9, 1))
        df = df_kmeans1_res1.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(9)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout);

    elif (z == 'house3'):
        km = KMeans(max(9, 1))
        df = df_kmeans1_res3.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(9)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house5'):
        km = KMeans(max(9, 1))
        df = df_kmeans1_res5.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(9)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house7'):
        km = KMeans(max(9, 1))
        df = df_kmeans1_res7.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(9)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house9'):
        km = KMeans(max(9, 1))
        df = df_kmeans1_res9.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(9)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house10'):
        km = KMeans(max(9, 1))
        df = df_kmeans1_res10.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(9)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
  
    
@app.callback(
    Output('cluster-graph2', 'figure'),
    [Input('x-variable1', 'value'),
    Input('y-variable1', 'value'),
    Input('radio1', 'value')])
def make_graph1(x, y, z):
    if (z == 'house1'):
        km = KMeans(max(3, 1))
        df = df_kmeans2_res1.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout);

    elif (z == 'house3'):
        km = KMeans(max(3, 1))
        df = df_kmeans2_res3.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house5'):
        km = KMeans(max(3, 1))
        df = df_kmeans2_res5.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house7'):
        km = KMeans(max(3, 1))
        df = df_kmeans2_res7.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house9'):
        km = KMeans(max(3, 1))
        df = df_kmeans2_res9.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house10'):
        km = KMeans(max(3, 1))
        df = df_kmeans2_res10.loc[:, [x, y]]
        km.fit(df.values)
        df["cluster"] = km.labels_

        data = [
            go.Scatter(
                x=df.loc[df.cluster == c, x],
                y=df.loc[df.cluster == c, y],
                mode="markers",
                marker={"size": 8},
                name="Cluster {}".format(c),
                    )
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)

@app.callback(
    Output('loadcurves', 'children'),
    [Input('radio2', 'value')]
    )
def loadcurves (value):
    if (value == 'load1'):
       return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res1.png')
            ])
    elif (value =='load2'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res2.png')
            ])
    elif (value =='load3'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res3.png')
            ])
    elif (value =='load4'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res4.png')
            ])
    elif (value =='load5'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res5.png')
            ])
    elif (value =='load6'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res6.png')
            ])
    elif (value =='load7'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res7.png')
            ])
    elif (value =='load8'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res8.png')
            ])
    elif (value =='load9'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res9.png')
            ])
    elif (value =='load10'):
        return html.Div([
            html.H6('Power load curve with 2 clusters'),
           html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, 
                    src='assets\loadcurve_res10.png')
            ])

