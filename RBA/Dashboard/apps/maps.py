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




# vancouver = gpd.read_file("assets/local-area-boundary.geojson")
df = pd.read_csv('assets/map_data.csv')
mean = pd.read_csv('assets/mean_consumption.csv')
# vc = gpd.read_file('assets/gpr_000b11a_e.shp')

df = pd.merge(df, mean,  left_on = df.index ,right_on= mean.index)


mapbox_access_token = 'pk.eyJ1IjoibGF2ZW5zaGkiLCJhIjoiY2txeTZnMjYzMTU2YTJxcWFwengydjhqbyJ9._T1SNgTdBRstJnPEdWra0A'
px.set_mapbox_access_token(mapbox_access_token)

# df['coordinates'] = df[['long','lat']].values.tolist()
# df['coordinates'] = df['coordinates'].apply(Point)

# df = gpd.GeoDataFrame(df, geometry = 'coordinates')
# print(vancouver)
# print(df)

# geo_house = pd.merge(vancouver, df, right_on= df.index, left_on = vancouver.index )
# # px.set_mapbox_access_token('pk.eyJ1IjoibGF2ZW5zaGkiLCJhIjoiY2txeTZnMjYzMTU2YTJxcWFwengydjhqbyJ9._T1SNgTdBRstJnPEdWra0A')

# figure = px.choropleth_mapbox(geo_house, geojson= vancouver.geometry,
#                               color = 
#                               zoom = 12, 
#                               locations = df['coordinates'],
#                               center = {"lat": 49.26205103 , "lon" : -123.1060941},
#                               range_color = [0,1000000000],
#                               mapbox_style = 'carto-positron')
# figure.update_layout (margin = {'r' : 0 , 't' : 0 , 'l' : 0 , 'b' : 0})



figure = px.scatter_mapbox(df, lat= 'lat', lon= 'long',
                        color = df['Mean_Consumption'] ,
                        hover_data = ['House'],
                        zoom = 12 , height = 500 ,
                        mapbox_style = 'carto-positron')




layout=html.Div(children=[
    html.Br(),
    html.Br(),
    dcc.Graph(
        
        id = 'map',
       
       figure = figure 
        
        )
    
    ])


 





        # fig = go.Figure(go.Scattermapbox(
        #         lat=['45.5017'],
        #         lon=['-73.5673'],
        #         mode='markers',
        #         marker=go.scattermapbox.Marker(
        #             size=14
        #         ),
        #         text=['Montreal'],
        #     ))
        
        # fig.update_layout(
        #     hovermode='closest',
        #     mapbox=dict(
        #         accesstoken=mapbox_access_token,
        #         bearing=0,
        #         center=go.layout.mapbox.Center(
        #             lat=45,
        #             lon=-73
        #         ),
        #         pitch=0,
        #         zoom=5
        #     )
        # )
        
        
    
        
 # fig, ax = plt.subplots(figsize = (10,8))
 #        sns.scatterplot(df['long'], df['lat'], marker = 'o', 
 #                        hue = df['House'], ax = ax)
 #        geo_df.plot(ax = ax, edgecolor = 'black')
        
        
 #        ax.legend(loc = 'center right', bbox_to_anchor=(1.7, 0.5), ncol=1) 
 #        ax.axis('off')
        
 #        plt.show()



# fig, ax = plt.subplots(1, figsize=(20,16)),
#         base = geo_df.plot(ax=ax),
#         df.plot(ax=base , column = 'House', marker="<", markersize = 15,
#                 cmap = 'cool', categorical = True , legend = True),
        
#         fig.show(



# print(geo_df)





