# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:29:12 2021

@author: Yara
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame
from mpl_toolkits import mplot3d

df = pd.read_csv('./Clean_data/res1.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'],format="%Y/%m/%d %H:%M:%S")
df = df.set_index('Date', drop=True)
df = df.interpolate() #interpolation was only done in this part to facilitate the use of teh following methods

########### Clustering ######################################################
## K-means Method
# Trial 1: Energy, temp, weekday and hour
cluster_df=df.drop(columns=['RelativeHumidity', 'WindSpeed10m_m/s', 'WindDirection10m', 'SurfacePressure_hPa', 'PrecipitableWater_kg/m2', 'GlobalSolarRad_W/m2', 'SnowDepth_LWE_cm']) 
#print(df.columns)

# create kmeans object
model = KMeans(n_clusters=9).fit(cluster_df) #define model and fit data to model
pred = model.labels_ 
#till now we don't know the best nb of clusters

#Find ideal cluster number: take range from 1 to 20 clusters
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc] #creating a set of objects (matrix of models) where number of clusters i changes 
score = [kmeans[i].fit(cluster_df).score(cluster_df) for i in range(len(kmeans))] #score is the square of distance of 1 point to the points inside, if point very far away from others, get very large value, if close, get small value
#this is one way to know ideal nb of clusters, sillhouet is another method we can use
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show() #should choose the point at the inflection, so here around 9 to 10 clusters would be good (so I can use the results from previous line)

#create a new column that presents the cluster number that the point belongs to
cluster_df['Cluster']=pred 

##Plots of the features in 2D (2 at a time)
# Temp vs Energy consumption
ax1=cluster_df.plot.scatter(x='Energy_res1',y='AirTemp_C', c='Cluster', colormap="Set1", sharex=False)
#We can notice that at low temperatures the energy consumption is not high but clustered at somehow low values. The conclusion that could be made here is that in Vancouver they used natural gas for heating, and thus the energy consumption does not increase on cold days
#some of the other clusters are intertwined (and stay like this for other cluster values)

#energy consumption vs hour
ax2=cluster_df.plot.scatter(x='Hour',y='Energy_res1',c='Cluster', colormap="Set1", sharex=False)
#Analysis: high values occur during the day, low values in morning, in evening highest values, visualization of how most power consumption is during mid day hours

#Day of week vs power
ax3=cluster_df.plot.scatter(x='Day of Week',y='Energy_res1',c='Cluster', colormap="Set1", sharex=False)
#this cluster is inconclusive because the clusters are all intertwined

#3D plot to see clusters of Power consumption over hours, weekdays 
fig = plt.figure()
ax = plt.axes(projection="3d")

cluster_0=cluster_df[pred==0]
cluster_1=cluster_df[pred==1]
cluster_2=cluster_df[pred==2]
cluster_3=cluster_df[pred==3]
cluster_4=cluster_df[pred==4]
cluster_4=cluster_df[pred==5]
cluster_4=cluster_df[pred==6]
cluster_4=cluster_df[pred==7]
cluster_4=cluster_df[pred==8]

cluster_0
ax.scatter3D(cluster_0['Hour'], cluster_0['Day of Week'],cluster_0['Energy_res1'],c='red');
ax.scatter3D(cluster_1['Hour'], cluster_1['Day of Week'],cluster_1['Energy_res1'],c='blue');
ax.scatter3D(cluster_2['Hour'], cluster_2['Day of Week'],cluster_2['Energy_res1'],c='green');

plt.show()


# Trial 2: Energy, RH, windspeed and solar radiation
cluster_df2 = df.drop(columns=['Day of Week', 'PrecipitableWater_kg/m2', 'Hour', 'SurfacePressure_hPa', 'PrecipitableWater_kg/m2', 'SnowDepth_LWE_cm', 'AirTemp_C', 'WindDirection10m']) 

# create kmeans object
model2 = KMeans(n_clusters=3).fit(cluster_df2) #define model and fit data to model
pred2 = model.labels_
#till now we don't know the best nb of clusters

#Find ideal cluster number: take range from 1 to 20 clusters
Nc2 = range(1, 20)
kmeans2 = [KMeans(n_clusters=i) for i in Nc2] #creating a set of objects (matrix of models) where number of clusters i changes 
score2 = [kmeans2[i].fit(cluster_df2).score(cluster_df2) for i in range(len(kmeans2))] #score is the square of distance of 1 point to the points inside, if point very far away from others, get very large value, if close, get small value #this is one way to know ideal nb of clusters, sillhouet is another method we can use
plt.plot(Nc2, score2)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show() #should choose the point at the inflection, so here around 2 to 3 clusters would be good

#create a new column that presents the cluster number that the point belongs to
cluster_df2['Cluster2']=pred2

##Plots of the features in 2D (2 at a time)
# RH vs Energy consumption
ax12=cluster_df2.plot.scatter(x='Energy_res1',y='RelativeHumidity', c='Cluster2', colormap="Set1", sharex=False)
#We can notice that at low temperatures the energy consumption is not high but clustered at somehow low values. The conclusion that could be made here is that in Vancouver they used natural gas for heating, and thus the energy consumption does not increase on cold days
#some of the other clusters are intertwined (and stay like this for other cluster values)

#energy consumption vs windspeed
ax22=cluster_df2.plot.scatter(x='Energy_res1',y='WindSpeed10m_m/s',c='Cluster2', colormap="Set1", sharex=False)
#Analysis: high values occur during the day, low values in morning, in evening highest values, visualization of how most power consumption is during mid day hours

#Solar Radiation vs power
ax32=cluster_df2.plot.scatter(x='GlobalSolarRad_W/m2',y='Energy_res1',c='Cluster2', colormap="Set1", sharex=False)
#this cluster is inconclusive because the clusters are all intertwined

