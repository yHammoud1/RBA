# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:51:08 2021

@author: Yara
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv('./Clean_data/res7.csv', parse_dates=['Date'])
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
ax1=cluster_df.plot.scatter(x='Energy_res7',y='AirTemp_C', c='Cluster', colormap="Set1", sharex=False)
#We can notice that at low temperatures the energy consumption is not high but clustered at somehow low values. The conclusion that could be made here is that in Vancouver they used natural gas for heating, and thus the energy consumption does not increase on cold days
#some of the other clusters are intertwined (and stay like this for other cluster values)

#energy consumption vs hour
ax2=cluster_df.plot.scatter(x='Hour',y='Energy_res7',c='Cluster', colormap="Set1", sharex=False)
#Analysis: high values occur during the day, low values in morning, in evening highest values, visualization of how most power consumption is during mid day hours

#Day of week vs power
ax3=cluster_df.plot.scatter(x='Day of Week',y='Energy_res7',c='Cluster', colormap="Set1", sharex=False)
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
ax.scatter3D(cluster_0['Hour'], cluster_0['Day of Week'],cluster_0['Energy_res7'],c='red');
ax.scatter3D(cluster_1['Hour'], cluster_1['Day of Week'],cluster_1['Energy_res7'],c='blue');
ax.scatter3D(cluster_2['Hour'], cluster_2['Day of Week'],cluster_2['Energy_res7'],c='green');

plt.show()


# Trial 2: Energy, RH, windspeed and solar radiation
cluster_df2 = df.drop(columns=['Day of Week', 'PrecipitableWater_kg/m2', 'Hour', 'SurfacePressure_hPa', 'PrecipitableWater_kg/m2', 'SnowDepth_LWE_cm', 'AirTemp_C', 'WindDirection10m']) 

# create kmeans object
model2 = KMeans(n_clusters=3).fit(cluster_df2) #define model and fit data to model
pred2 = model2.labels_
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
ax12=cluster_df2.plot.scatter(x='Energy_res7',y='RelativeHumidity', c='Cluster2', colormap="Set1", sharex=False)
#at low RH values, very low energy consumption occurs while at high RH, higher energy values are seen

#energy consumption vs windspeed
ax22=cluster_df2.plot.scatter(x='Energy_res7',y='WindSpeed10m_m/s',c='Cluster2', colormap="Set1", sharex=False)
#there is no clear correlation

#Solar Radiation vs energy consumption
ax32=cluster_df2.plot.scatter(x='GlobalSolarRad_W/m2',y='Energy_res7',c='Cluster2', colormap="Set1", sharex=False)
#at low solar radiation range, the energy values are higher than at higher solar radiation

# Trial 3: Energy, Snow depth, Precipitable water and wind direction
cluster_df3 = df.drop(columns=['Day of Week', 'GlobalSolarRad_W/m2', 'Hour', 'SurfacePressure_hPa', 'RelativeHumidity', 'AirTemp_C', 'WindSpeed10m_m/s', 'AirTemp_C']) 

# create kmeans object
model3 = KMeans(n_clusters=4).fit(cluster_df3) #define model and fit data to model
pred3 = model3.labels_
#till now we don't know the best nb of clusters

#Find ideal cluster number: take range from 1 to 20 clusters
Nc3 = range(1, 20)
kmeans3 = [KMeans(n_clusters=i) for i in Nc3] #creating a set of objects (matrix of models) where number of clusters i changes 
score3 = [kmeans3[i].fit(cluster_df3).score(cluster_df3) for i in range(len(kmeans3))] #score is the square of distance of 1 point to the points inside, if point very far away from others, get very large value, if close, get small value #this is one way to know ideal nb of clusters, sillhouet is another method we can use
fig, ax7 = plt.subplots()
ax7.plot(Nc3, score3)
ax7.set(xlabel='Number of Clusters', ylabel='Score')
ax7.set_title('Elbow Curve')
#should choose the point at the inflection, so here around 3 to 4 clusters would be good

#create a new column that presents the cluster number that the point belongs to
cluster_df3['Cluster3']=pred3

##Plots of the features in 2D (2 at a time)
# Rain vs Energy consumption
ax13=cluster_df3.plot.scatter(x='Energy_res7',y='PrecipitableWater_kg/m2', c='Cluster3', colormap="Set1", sharex=False)
#no clear correlation, although 1 cluster is clear at low precipitation values

#energy consumption vs snow depth
ax23=cluster_df3.plot.scatter(x='SnowDepth_LWE_cm',y='Energy_res7',c='Cluster3', colormap="Set1", sharex=False)
#no clear correlation

#Wind direction vs energy consumption
ax33=cluster_df3.plot.scatter(x='WindDirection10m',y='Energy_res7',c='Cluster3', colormap="Set1", sharex=False)
#at any wind direction range, the energy values vary from min to max