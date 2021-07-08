# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 18:37:43 2021

@author: Yara
"""
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from app import app
import pandas as pd
import  plotly.express as px

table = pd.read_csv('house_charact.csv')

table1 = dbc.Table.from_dataframe(table,bordered= True, dark= True, hover = True,responsive= True, striped= True )


