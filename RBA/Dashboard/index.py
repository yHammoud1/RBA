import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

from app import app
from apps import raw_energy_data, eda, clustering, feature_engineering, forecast_models, maps , best_forecast_model, home






SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15rem",
    "padding": "2rem 1rem",
    "background-color": "#E2C2FF"
    
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("----", className="display-4"),
        html.Hr(),
        dbc.Nav( 
            [
                dbc.NavLink("Home", href="/home", active="exact"),
                dbc.NavLink("Raw Energy Data", href="/raw_energy_data", active="exact"),
                dbc.NavLink("Study Area", href="/maps", active="exact"),
                dbc.NavLink("Exploratory Data Analysis", href="/eda", active="exact"),
                dbc.NavLink("Clustering", href="/clustering", active="exact"),
                dbc.NavLink("Feature Engineering", href="/feature_engineering", active="exact"),
                dbc.NavLink("Forecast Models", href="/forecast_models", active="exact"),
                dbc.NavLink("Best Forecast Model", href="/best_forecast_model", active="exact"),
                
            ],
            vertical=True,
            pills=True,
            
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)



app.layout =   html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
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
    elif pathname == '/forecast_models':
      return forecast_models.layout
    elif pathname == '/best_forecast_model':
        return best_forecast_model.layout
    elif pathname == '/maps':
        return maps.layout
    else:
        return  home.layout
    
    
       

if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_props_check=False)
    
  
    
    
    




