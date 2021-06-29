# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:56:50 2021

@author: Yara
"""
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


df = pd.read_csv('Proj3_clean_data.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'],format="%Y/%m/%d %H:%M:%S")
df = df.set_index('Date', drop=True)

########### Display Data ################################################################
fig1, ax1 = plt.subplots()
ax1.plot(df.index, df['AirTemp_C'])
#ax1.set_title('A single plot')

fig2, ax2 = plt.subplots()
ax2.plot(df.index, df['Energy_res1'])

#plt.plot( df['AirTemp_C'])
#plt.plot( df['Energy_res1'])
