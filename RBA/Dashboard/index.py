# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:11:38 2021

@author: Yara
"""
import pandas as pd
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from apps import raw_energy_data, home, eda, clustering, feature_engineering, maps 
# forecast_models, best_forecast_model

data = [['House1', 49.26980201, -123.0837633],
        ['House2', 49.26781432, -123.0674411]]

df = pd.DataFrame(data, columns = ['House', 'long', 'lat'])

app.layout = html.Div(children=[
        dcc.Location(id="url"),
        dbc.NavbarSimple(
            children=[
                dbc.NavLink("Home", href="/home", active="exact"),
                dbc.NavLink("Raw Energy Data", href="/raw_energy_data", active="exact"),
                dbc.NavLink("EDA", href="/eda", active="exact"),
                dbc.NavLink("Clustering", href="/clustering", active="exact"),
                dbc.NavLink("Feature Engineering", href="/feature_engineering", active="exact"),
                dbc.NavLink("Forecast Models", href="/forecast_models", active="exact"),
                dbc.NavLink("Best Forecast Model", href="/best_forecast_model", active="exact"),
                dbc.NavLink("Map", href="/maps", active="exact"),
            ],
            brand="IST South Tower 2017-2018",
            color="info",
            dark=True,
            fixed= "top",
            
        ),
       
        dbc.Container(
            id="page-content", className="pt-4",  
                      
                      ),
        ])



@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/raw_energy_data':
        return raw_energy_data.layout
    elif pathname == '/eda':
         return eda.layout
    elif pathname == '/clustering':
        return clustering.layout
    elif pathname == '/feature_engineering':
        return feature_engineering.layout
    #elif pathname == '/forecast_models':
    #   return forecast_models.layout
    # elif pathname == '/best_forecast_model':
    #     return best_forecast_model.layout
    elif pathname == '/maps':
        return html.Div( children = [
            html.Br(),
            html.Br(),
            dcc.Graph(
                id='example-graph4',
                figure = px.scatter_mapbox(df, lat="lat", lon="long",
                                        color_discrete_sequence=["fuchsia"], zoom=10, height=400,mapbox_style="open-street-map"))

            ])
    else:
        return home.layout
    
    
       

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)