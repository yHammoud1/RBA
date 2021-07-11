# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 15:36:28 2021

@author: laven
"""

import pandas as pd
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from geopandas import GeoDataFrame
import json
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from shapely.geometry import Point 
import plotly.graph_objects as go
from app import app 


from plotly.tools import mpl_to_plotly





df = pd.read_csv('assets/map_data.csv')
mean = pd.read_csv('assets/mean_consumption.csv')


df = pd.merge(df, mean,  left_on = df.index ,right_on= mean.index)


mapbox_access_token = 'pk.eyJ1IjoibGF2ZW5zaGkiLCJhIjoiY2txeTZnMjYzMTU2YTJxcWFwengydjhqbyJ9._T1SNgTdBRstJnPEdWra0A'
px.set_mapbox_access_token(mapbox_access_token)



figure = px.scatter_mapbox(df, lat= 'lat', lon= 'long',
                        color = df['Mean_Consumption'] ,
                        hover_data = ['House'],
                        zoom = 12 , height = 500 ,
                        mapbox_style = 'streets')



layout=html.Div(children=[
    html.Br(),
    html.H4('Map of the Vancouver area showing the geographic locations of the houses included in teh study.'),
    
    html.H5('The mean energy consumption over the 3 years of study was calculated for each house and are presented in the map varying teh consumption by color.'),
    html.Br(),
    dcc.Graph(
        
        id = 'map',
       
       figure = figure 
        
        )
    
    ])





