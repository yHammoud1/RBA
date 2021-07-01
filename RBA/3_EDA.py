# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 22:38:48 2021

@author: Yara
"""
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sb


df = pd.read_csv('Proj3_clean_data.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'],format="%Y/%m/%d %H:%M:%S")
df = df.set_index('Date', drop=True)

########### Visual Check for Outliers ###################################################
#was done by sorting each variable using variable explorer in spyder
#Findings in Energy conusmption data
    #Residential 1: two zero values found in the energy consumption: during 2 consecutive hours on 17-11-2015 (which is expected to be abnormal)
    #Residential 2: no apparent outliers 
    #Residential 3 & 4 & 7: zero values seen for some hours in 2 consecutive days for both these houses
    #Residential 5: two zero values found in the energy consumption: during 2 consecutive hours on 01-01-2017
    #Residential 6: no apparent outliers
    #Residential 8: four zero values (common with res 3 4 and 7)
    #Residential 9: few zero values
    #Residential 10: one zero value
#Findings in meteo data: no apparent outliers

#Box plots
for i in df.columns:
    plt.figure()
    sb.boxplot(x=df[i])
    
    