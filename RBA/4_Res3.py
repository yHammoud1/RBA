# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:37:14 2021

@author: Yara
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest #selection method
from sklearn.feature_selection import f_regression, mutual_info_regression #score matrix
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('./Clean_data/res3.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'],format="%Y/%m/%d %H:%M:%S")
df = df.set_index('Date', drop=True)
df = df.interpolate() #interpolation was only done in this part to facilitate the use of teh following methods

########### Clustering ######################################################
## K-means Method
# Trial 1: Energy, temp, weekday and hour
cluster_df=df.drop(columns=['RelativeHumidity', 'WindSpeed10m_m/s', 'WindDirection10m', 'SurfacePressure_hPa', 'PrecipitableWater_kg/m2', 'GlobalSolarRad_W/m2', 'SnowDepth_LWE_cm']) 
#print(df.columns)

# create kmeans object
model = KMeans(n_clusters=8).fit(cluster_df) #define model and fit data to model
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
plt.show() #should choose the point at the inflection, so here around 8 to 9 clusters would be good (so I can use the results from previous line)

#create a new column that presents the cluster number that the point belongs to
cluster_df['Cluster']=pred 

##Plots of the features in 2D (2 at a time)
# Temp vs Energy consumption
ax1=cluster_df.plot.scatter(x='Energy_res3',y='AirTemp_C', c='Cluster', colormap="Set1", sharex=False)
#We can notice that at low temperatures the energy consumption is not high but clustered at somehow low values. The conclusion that could be made here is that in Vancouver they used natural gas for heating, and thus the energy consumption does not increase on cold days
#some of the other clusters are intertwined (and stay like this for other cluster values)

#energy consumption vs hour
ax2=cluster_df.plot.scatter(x='Hour',y='Energy_res3',c='Cluster', colormap="Set1", sharex=False)
#Analysis: high values occur during the day, low values in morning, in evening highest values, visualization of how most power consumption is during mid day hours

#Day of week vs power
ax3=cluster_df.plot.scatter(x='Day of Week',y='Energy_res3',c='Cluster', colormap="Set1", sharex=False)
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
ax.scatter3D(cluster_0['Hour'], cluster_0['Day of Week'],cluster_0['Energy_res3'],c='red');
ax.scatter3D(cluster_1['Hour'], cluster_1['Day of Week'],cluster_1['Energy_res3'],c='blue');
ax.scatter3D(cluster_2['Hour'], cluster_2['Day of Week'],cluster_2['Energy_res3'],c='green');

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
ax12=cluster_df2.plot.scatter(x='Energy_res3',y='RelativeHumidity', c='Cluster2', colormap="Set1", sharex=False)
#at low RH values, very low energy consumption occurs while at high RH, higher energy values are seen

#energy consumption vs windspeed
ax22=cluster_df2.plot.scatter(x='Energy_res3',y='WindSpeed10m_m/s',c='Cluster2', colormap="Set1", sharex=False)
#there is no clear correlation

#Solar Radiation vs energy consumption
ax32=cluster_df2.plot.scatter(x='GlobalSolarRad_W/m2',y='Energy_res3',c='Cluster2', colormap="Set1", sharex=False)
#at any solar radiation range, the energy values vary from min to max

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
ax13=cluster_df3.plot.scatter(x='Energy_res3',y='PrecipitableWater_kg/m2', c='Cluster3', colormap="Set1", sharex=False)
#no clear correlation, although 1 cluster is clear at low precipitation values

#energy consumption vs snow depth
ax23=cluster_df3.plot.scatter(x='SnowDepth_LWE_cm',y='Energy_res3',c='Cluster3', colormap="Set1", sharex=False)
#no clear correlation

#Wind direction vs energy consumption
ax33=cluster_df3.plot.scatter(x='WindDirection10m',y='Energy_res3',c='Cluster3', colormap="Set1", sharex=False)
#at any wind direction range, the energy values vary from min to max



########### Identifying Daily Patterns ######################################################
cluster_df['date only'] = [d.date() for d in cluster_df.index]
cluster_df['date only'] = pd.to_datetime(cluster_df['date only'], format='%Y-%m-%d')
cluster_df = cluster_df.set_index('date only', drop = True)


cluster_df = cluster_df.drop(columns=['AirTemp_C','Day of Week','Cluster'])

##Create a pivot table needed to plot the load curve
df_pivot = cluster_df.groupby(by=['date only', 'Hour']).Energy_res3.sum().unstack()
df_pivot = df_pivot.dropna()
#pivot table: represent power as an array over the days --> for each day I have the power points over the hours of day

##Load curve plot
df_pivot.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.02)
plt.ylim([0, 6])
plt.show()
#can see around 3 clusters of data: days with very high consumption (probably monday through wednesday), days with medium consumption (probably thurs and fri)
#and finally weekend days are seen in a cluster alone as well (the flat curves below)

##Clustering of this data
sillhoute_scores = []
n_cluster_list = np.arange(2,10).astype(int)

X = df_pivot.values.copy()
    
# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    
    kmeans4 = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans4.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans4.labels_))
    
plt.plot(n_cluster_list,sillhoute_scores)
plt.show()
#graph shows that maybe best is 2 clusters --> will continue with 2 clusters

#create two indices for the pivot table: by adding a cluster one
kmeans4 = KMeans(n_clusters=2)
cluster_found = kmeans4.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='blabla')
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )

fig, ax6 = plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_pivot.index.get_level_values('blabla').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot.xs(cluster, level=1).T.plot(
        ax=ax6, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}'
        )
    df_pivot.xs(cluster, level=1).median().plot(
        ax=ax6, color=color, alpha=0.9, ls='--'
    )

ax6.set_xticks(np.arange(1,25))
ax6.set_ylabel('KWh')
ax6.set_xlabel('Hour')
plt.ylim([0, 6])
plt.show()
#can clearly see the two clusters, these could be weekends and weekdays, or due to daylight saving but this cannot be so clear from the plot



########### Feature Selection ######################################################
#create new column called Energy-1: the energy of the previous hour
df['Energy-1'] = df['Energy_res3'].shift(1)
df = df.drop([df.index[0]])

#change dataframes to matrices
# Define input and outputs
X=df.values

Y=X[:,0] #output of my model is column 2 (Energy)
X=X[:,[1,2,3,4,5,6,7,8,9,10,11]] #all the other columns are input --> put them all as X


#### Filter Methods
### kBest
#in this method I will calc correlation score, and find the best features
 
##Selecting 2 best features: f-test ANOVA
features=SelectKBest(k=2,score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #calculates the f_regression between all the features and the output
print(fit.scores_)
features_results=fit.transform(X) #this will tell me what are the best 2 features
print(features_results)
#the results show that the 2 most relevant features in order are the Energy-1 and Hour

##Selecting 3 best features: f-test ANOVA
features3=SelectKBest(k=3,score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit3=features3.fit(X,Y) #calculates the f_regression between all the features and the output
print(fit3.scores_)
features_results3=fit3.transform(X) #this will tell me what are the best 2 features
print(features_results3)
#the results show that the 3 most relevant features in order are the energy-1, Hour, and Snow depth (LWE)
#from the results we can see that the 4th best one would be Solar radiation
#least relevant one is the wind speed


##Selecting 2 best features: Mutual_info_regression
features=SelectKBest(k=2,score_func=mutual_info_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #calculates the f_regression between all the features and the output
print(fit.scores_)
features_results=fit.transform(X) #this will tell me what are the best 2 features
print(features_results)
#2 best features are Energy-1 and hour

##Selecting 3 best features: mutual_info_regression
features=SelectKBest(k=3,score_func=mutual_info_regression)
fit=features.fit(X,Y) #calculates the f_regression of the features, calculating the correlation
print(fit.scores_)
features_results=fit.transform(X) #which are the best 3 features
print(features_results) 
#3 best features are Power-1, hour and precipitable water

#### Wrapper Methods
### Recursive Feature Elemination
model=LinearRegression() # LinearRegression Model as Estimator
rfe=RFE(model,n_features_to_select=2)# using 2 features
rfe2=RFE(model,n_features_to_select=3) # using 3 features
rfe3=RFE(model,n_features_to_select=3) # using 3 features
fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)
fit3=rfe3.fit(X,Y)

print( "Feature Ranking (Linear Model, 2 features): %s" % (fit.ranking_)) #Surface pressure and precipitable water features
print( "Feature Ranking (Linear Model, 3 features): %s" % (fit2.ranking_)) #Snow Depth, Solar rad and wind direction features
print( "Feature Ranking (Linear Model, 3 features): %s" % (fit3.ranking_)) #Snow Depth, Solar rad and wind direction features

#the results of this method are quite different than the previous ones, and not logical as much

### Ensemble Method
model = RandomForestRegressor() 
model.fit(X, Y)
print(model.feature_importances_)
#Results show: Energy-1, hour, air temp



########### Feature Extraction/Engineering #######################################
#Note that holidays are included in the day of week column, with the numbe 7
#feature tools for automated feature engineering

#Square of temperature: 10 degrees difference in temp during the day but also 10 degrees btwn summer and winter which is low --> atrificially inc the diff btwn high and low temp: useful to differentitate winter and summer
df['temp2']=np.square(df['AirTemp_C'])

#Square of snow Depth (or LWE): could be useful to differentiate winter and summer by artificially increasing the differences
df['LWE2']=np.square(df['SnowDepth_LWE_cm'])

#Precipitation square --> higher diff btwn days without rain and days with rain
df['Prec2']=np.square(df['PrecipitableWater_kg/m2'])

# Heating degree.hour
df['HDH']=np.maximum(0,df['AirTemp_C']-16) #max btwn temp and reference value (16 degrees, and 0)

# recurrent: working with Power-1 as missing now
#will have to interpolate here because the models do not work with nan values
df = df.interpolate()
X=df.values
Y=X[:,11]
X=X[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15]]
model = RandomForestRegressor()
model.fit(X, Y)
print(model.feature_importances_) # 2 most relevant features: Energy and Hour 


#final decision of features: Energy-1, Solar radiation, Temp, Hour, day of week
df_model = df.drop(columns=['SnowDepth_LWE_cm', 'RelativeHumidity', 'SurfacePressure_hPa', 'WindDirection10m', 'PrecipitableWater_kg/m2', 'WindSpeed10m_m/s', 'temp2', 'LWE2', 'Prec2', 'HDH'])


########## Regression Models #################################################
## Split training and testing data
#split according to several features
X=df_model.values
Y=X[:,0] #output is power
X=X[:,[1,2,3,4,5]] #input are all the other columns

from sklearn.model_selection import train_test_split
#Split randomly
#by default, it chooses randomly 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,Y)



### Linear Regression
from sklearn import  linear_model

# Create linear regression object
LR_model = linear_model.LinearRegression()

# Train the model using the training sets
LR_model.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = LR_model.predict(X_test)

plt.plot(y_test[1:200], label='testing')
plt.plot(y_pred_LR[1:200], label='predicted')
plt.legend()
plt.show()

plt.scatter(y_test,y_pred_LR)
#we can see that the predicted data always looks a bit underfit

from sklearn import  metrics #MAE, RMSE, ...
#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print(MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR)



### Support Vector Machine
from sklearn.preprocessing import StandardScaler #to normalize the data bcz this method is so sensitive to the values of data
#standardscaler: uses the standard normal distribution
from sklearn.svm import SVR

sc_X = StandardScaler() #for transforming the data
sc_y = StandardScaler()
X_train_SVR = sc_X.fit_transform(X_train)
y_train_SVR = sc_y.fit_transform(y_train.reshape(-1,1)) #reshape is used bcz line vector becomes column vector

#values close to 0 are avg consumption, between -1 and 1 are t=within the std dev from avg, others are very high or very low values

model_SVR = SVR(kernel='rbf') #rbf: transforms the data using exponential function
model_SVR.fit(X_train_SVR,y_train_SVR)

y_pred_SVR = model_SVR.predict(sc_X.fit_transform(X_test))
y_test_SVR=sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=sc_y.inverse_transform(y_pred_SVR) #transform back to my scale to understand it
#y_pred_SVR = sc_y.inverse_transform(regr.predict(sc_X.fit_transform(X_test)))
plt.plot(y_test_SVR[1:200], label='testing')
plt.plot(y_pred_SVR[1:200], label='predicted')
plt.legend()
plt.show()
plt.plot(y_test[1:200], label='testing')
plt.plot(y_pred_SVR2[1:200], label='predicted')
plt.legend()

MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)
print(MAE_SVR, MSE_SVR, RMSE_SVR,cvRMSE_SVR)




### Regression Desicion Tree
from sklearn.tree import DecisionTreeRegressor
# Create Regression Decision Tree object
DT_model= DecisionTreeRegressor()

# Train the model using the training sets
DT_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = DT_model.predict(X_test)

plt.plot(y_test[1:200], label='tested')
plt.plot(y_pred_DT[1:200], label='predicted')
plt.legend()
plt.show()
plt.scatter(y_test,y_pred_DT)

#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
print(MAE_DT, MSE_DT, RMSE_DT,cvRMSE_DT)



### Random Forest
from sklearn.ensemble import RandomForestRegressor
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)

plt.plot(y_test[1:200], label='tested')
plt.plot(y_pred_RF[1:200], label='predicted')
plt.legend()
plt.show()
plt.scatter(y_test,y_pred_RF)



### Uniformized Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Random Forest Uniformized Data
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}

RFU_model = RandomForestRegressor(**parameters)
RFU_model.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RFU = RFU_model.predict(X_test_scaled)

#Evaluate errors
MAE_RFU=metrics.mean_absolute_error(y_test,y_pred_RFU) 
MSE_RFU=metrics.mean_squared_error(y_test,y_pred_RFU)  
RMSE_RFU= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RFU))
cvRMSE_RFU=RMSE_RFU/np.mean(y_test)
print(MAE_RFU,MSE_RFU,RMSE_RFU,cvRMSE_RFU)

plt.plot(y_test[1:200], label='tested')
plt.plot(y_pred_RFU[1:200], label='predicted')
plt.legend()
plt.show()
plt.scatter(y_test,y_pred_RFU)



### Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#GB_model = GradientBoostingRegressor(**params)

GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
print(MAE_GB,MSE_GB,RMSE_GB,cvRMSE_GB)

plt.plot(y_test[1:200], label='tested')
plt.plot(y_pred_GB[1:200], label='predicted')
plt.legend()
plt.show()
plt.scatter(y_test,y_pred_GB)



### Extreme Gradient Boosting
from xgboost import XGBRegressor

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#GB_model = GradientBoostingRegressor(**params)

XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)

MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print(MAE_XGB,MSE_XGB,RMSE_XGB,cvRMSE_XGB)

plt.plot(y_test[1:200], label='tested')
plt.plot(y_pred_XGB[1:200], label='predicted')
plt.legend()
plt.show()
plt.scatter(y_test,y_pred_XGB)



### Bootstraping
from sklearn.ensemble import BaggingRegressor

BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)

MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
print(MAE_BT,MSE_BT,RMSE_BT,cvRMSE_BT)

plt.plot(y_test[1:200], label='tested')
plt.plot(y_pred_BT[1:200], label='predicted')
plt.legend()
plt.show()
plt.scatter(y_test,y_pred_BT)




### Neural Networks
from sklearn.neural_network import MLPRegressor

NN_model = MLPRegressor(hidden_layer_sizes=(5,5)) # we can change the hidden layer size to try which is better
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)

MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
print(MAE_NN,MSE_NN,RMSE_NN,cvRMSE_NN)

plt.plot(y_test[1:200], label='tested')
plt.plot(y_pred_NN[1:200], label='predicted')
plt.legend()
plt.show()
plt.scatter(y_test,y_pred_NN)


######## Final Model: Random Forest #######################################
# Random forest and gradient boosing gave the best results 
# --> will continue wil some trials to improve the results from random forest

## Trial 1: min_sample_leaf= 6  , n_estimators= 100
parameters = {'bootstrap': True,
              'min_samples_leaf': 6,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)


## Trial 2: min_sample_leaf= 3  , n_estimators= 150, min_samples_split= 30 
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 150, 
              'min_samples_split': 30,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)

## Trial 3: min_sample_leaf= 3  , n_estimators= 300, min_samples_split= 10, max_depth= 40 
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 300, 
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 40,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)


# Final Comments
# It is important to highlight that some missing data were interpolated in order to fill the gap in data because for some of the machine learning models, it was not feasible to have nan values 
# This might be a vital factor that moderates the quality of the model (all models in general are highly affected by the data quality)
# and could be the reason for a cvRMSE that is not very low, respectively. 
#Other interpolation methods could be used, or as mentioned at the beginning, the data could be left as nan and then finally predicted by the model along with the rest of the values that the method works to produce

