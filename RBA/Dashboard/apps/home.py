import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from app import app
import pandas as pd
import  plotly.express as px

table = pd.read_csv('house_charact.csv')

table1 = dbc.Table.from_dataframe(table,bordered= True, dark= False, hover = True,responsive= True, striped= True )


layout= html.Div(children=[
    html.Br(),
    html.H2('Vancouver Residential Buildings Energy Consumption Analysis'),
    html.Br(),
    html.H5('This dashboard presents an energy consumption analysis done on 10 residential buildings in the area of Vancouver, British Columbia, Canada. All the houses included are powered by BC hydro, which povides nearly 100% hydro-electric power.'),
    html.H5('Hourly energy consumption data, amongst other data about the houses under study, were taken from a dataset collected from a research project done at Harvard University (Makonin, S., 2018. HUE: The hourly usage of energy dataset for buildings in British Columbia).'),
    html.H5('The meteo data for the same area were taken from an open online source: https://toolkit.solcast.com.au/'), 
    html.H6('The analysis is done over a period of 3 years from 2015 to 2018, where hourly data for all houses is nearly fully available.'), 
    
    html.Br(),
    html.H6('The table below presents some of the characteristics of the houses under study.'),
    html.H6('House types include:'),
    html.H6('character   - multi-level houses build before 1940'),
    html.H6('special     - two-level houses built between 1965 to 1989'),
    html.H6('modern      - two-/three-level houses build in the 1990s and afterwards'),
    html.H6('duplex      - two houses that share a common wall, can be side-by-side or front-back'),
    html.H6('apartment   - hight-rise or low-rise living units.'),
    html.Br(),
    html.H6('The column cover refers to data coverage (ie. the percentage of non-missing data is shown.'),
    html.H6('Facing means what direction the house is facing. This often has an impact on house cooling durning the summer. East and West facing houses get hotter faster.'),
    html.Div(table1),
    html.H6('Discussion here...')
    ])
