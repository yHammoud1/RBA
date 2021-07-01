# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:56:50 2021

@author: Yara
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Proj3_clean_data.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'],format="%Y/%m/%d %H:%M:%S")
df = df.set_index('Date', drop=True)

########### Display Data ################################################################

#fig1, ax1 = plt.subplots()
#ax1.plot(df.index, df['AirTemp_C'])
#ax1.set_title('A single plot')

for i in df.columns:
    plt.figure()
    plt.plot(df[i])
    plt.title(i)

########### Basic Statistics #############################################################
#Histograms
for i in df.columns:
    plt.figure()
    plt.hist(df[i], 50)
    plt.title(i)
    
#Scatter plots
plt.figure('A')
plt.scatter(df['Energy_res1'],df['AirTemp_C'])
plt.title('Energy res1 vs Air temp')

plt.figure('B')
plt.scatter(df['Energy_res2'],df['AirTemp_C']) 
plt.title('Energy res2 vs Air temp')

plt.figure('C')
plt.scatter(df['Energy_res2'],df['PrecipitableWater_kg/m2']) 
plt.title('Energy res2 vs Precipitable water kg/m2')
    
for i in df.columns:
    x = np.mean(df[i])
    print(i + ': ' + str(x))

#we can notice that the energy consumption is highest for residentials 2 and 10, with 2 having 
# a 'character' house type which means it is a multi-level old house, which is the main reason
# for it having such a high consumption. In addition, residential 10 is a modern 2-3 level house
# which also explains the high consumption.
#On teh other hand the two residentials with lowest consumption are 4 and 8, both being simple
# small apartments which is also the reason behind the low consumption
#We can notice also that the average air temperature in Vancouver is around 8 C, which is relatively 
# low and leads to usage of a lot of heating. 

#Calculate Correlation
correlation=df.corr('spearman') #check what does spearman do here
print(correlation)


