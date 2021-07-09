import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from app import app

import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from dash.dependencies import Input, Output

df_kmeans1_res1= pd.read_csv('../../Result_csv/Cluster_kmeans1_res1.csv')
df_kmeans1_res3= pd.read_csv('../../Result_csv/Cluster_kmeans1_res3.csv')
df_kmeans1_res5= pd.read_csv('../../Result_csv/Cluster_kmeans1_res5.csv')
df_kmeans1_res7= pd.read_csv('../../Result_csv/Cluster_kmeans1_res7.csv')
df_kmeans1_res9= pd.read_csv('../../Result_csv/Cluster_kmeans1_res9.csv')
df_kmeans1_res10= pd.read_csv('../../Result_csv/Cluster_kmeans1_res10.csv')

df_kmeans2_res1= pd.read_csv('../../Result_csv/Cluster_kmeans2_res1.csv')
df_kmeans2_res3= pd.read_csv('../../Result_csv/Cluster_kmeans2_res3.csv')
df_kmeans2_res5= pd.read_csv('../../Result_csv/Cluster_kmeans2_res5.csv')
df_kmeans2_res7= pd.read_csv('../../Result_csv/Cluster_kmeans2_res7.csv')
df_kmeans2_res9= pd.read_csv('../../Result_csv/Cluster_kmeans2_res9.csv')
df_kmeans2_res10= pd.read_csv('../../Result_csv/Cluster_kmeans2_res10.csv')

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x-variable",
                    options=[
                        {"label": 'Energy', "value":'Energy_res' },
                        {"label": 'Hour', "value":'Hour' }, 
                        {"label": 'Week Day', "value":'Week Day' } 
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
                        {"label": 'Temperature', "value": 'temp_C' },
                        {"label": 'Energy', "value":'Energy_res' }
                    ],
                    value="temp_C",
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
                        {"label": 'Relative Humidity', "value": 'HR' },
                        {"label": 'Wind Speed', "value":'windspped' },
                        {"label": 'Solar Radiation', "value":'solarRad' },
                    ],
                    value="HR",
                ),
            ]
        ),
        
    ],
    body=True,
) 


layout=html.Div(children=[
    html.Br(), 
    html.Br(), 
    html.H4('Clustering of Data Using Kmeans and Clustered Load Curve Results'),
    dcc.Tabs(id='tabs-1', value='tab-1', children=[
         dcc.Tab(label='Kmeans', value='tab-1', id='kmeanstab', children=[
             html.Div([
       html.H5('Clustering using Kmeans: trials with sets of different parameters'),
       html.H6('Optimal number of clusters --'),
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
    value='house1',
    labelStyle={'display': 'inline-block'}
    ), 
            
        html.H6('Trial 1: Energy, Temperature, Weekday and Hour'),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(dcc.Graph(id="cluster-graph"), md=8),
            ],
            align="center",
                ),
        html.H6('Trial 2: Energy, Humidity, Wind speed and Solar radiation'),
                
        dbc.Row(
            [
                dbc.Col(controls1, md=4),
                dbc.Col(dcc.Graph(id="cluster-graph2"), md=8),
            ],
            align="center",
                ),
        ])
             ]),
         
         
         dcc.Tab(label='Load Curve', value='tab-2'),
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
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout);

    elif (z == 'house3'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house5'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house7'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house9'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house10'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
  
    
@app.callback(
    Output('cluster-graph2', 'figure'),
    [Input('x-variable1', 'value'),
    Input('y-variable1', 'value')])
def make_graph1(x, y, z):
    if (z == 'house1'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout);

    elif (z == 'house3'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house5'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house7'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house9'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)
    
    elif (z == 'house10'):
        km = KMeans(max(3, 1))
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
            for c in range(3)
            ]
        layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

        return go.Figure(data=data, layout=layout)


