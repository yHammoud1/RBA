# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:11:38 2021

@author: Yara
"""
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app
from apps import raw_energy_data
#eda, clustering, feature_engineering, forecast_models, best_forecast_model



app.layout = html.Div(children=[
        dcc.Location(id="url"),
        dbc.NavbarSimple(
            children=[
                dbc.NavLink("Raw Energy Data", href="/raw_energy_data", active="exact"),
                dbc.NavLink("EDA", href="/eda", active="exact"),
                dbc.NavLink("Clustering", href="/clustering", active="exact"),
                dbc.NavLink("Feature Engineering", href="/feature_engineering", active="exact"),
                dbc.NavLink("Forecast Models", href="/forecast_models", active="exact"),
                dbc.NavLink("Best Forecast Model", href="/best_forecast_model", active="exact"),
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
    # elif pathname == '/eda':
    #     return eda.layout
    # elif pathname == '/clustering':
    #     return clustering.layout
    # elif pathname == '/feature_engineering':
    #     return feature_engineering.layout
    # elif pathname == '/forecast_models':
    #     return forecast_models.layout
    # elif pathname == '/best_forecast_model':
    #     return best_forecast_model.layout
    else:
        return html.Img(style={'display': 'block','margin-left': 'auto','margin-right': 'auto'}, src='assets\main.png'),
       


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)