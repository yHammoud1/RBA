# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 22:38:48 2021

@author: Yara
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from functools import reduce


df = pd.read_csv('Proj3_clean_data_combined.csv', parse_dates=['Date'])
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
    
########### Removing Outliers ###################################################

## Z-score Method: (note that from the histograms of energy data we notice that they are not normally distributed so the zscore method will not be so effecitve/accurate)
#Calculate the zscore of the energy consumption data of all residentials in absolute value
#z = np.abs(stats.zscore(df['Energy_res1']))
#I had to use another code to calculate the zscore because the new one manages better the presence of nan values in the data

z1 = np.abs((df['Energy_res1'] - df['Energy_res1'].mean())/df['Energy_res1'].std(ddof=0))
z2 = np.abs((df['Energy_res2'] - df['Energy_res2'].mean())/df['Energy_res2'].std(ddof=0))
z3 = np.abs((df['Energy_res3'] - df['Energy_res3'].mean())/df['Energy_res3'].std(ddof=0))
z4 = np.abs((df['Energy_res4'] - df['Energy_res4'].mean())/df['Energy_res4'].std(ddof=0))
z5 = np.abs((df['Energy_res5'] - df['Energy_res5'].mean())/df['Energy_res5'].std(ddof=0))
z6 = np.abs((df['Energy_res6'] - df['Energy_res6'].mean())/df['Energy_res6'].std(ddof=0))
z7 = np.abs((df['Energy_res7'] - df['Energy_res7'].mean())/df['Energy_res7'].std(ddof=0))
z8 = np.abs((df['Energy_res8'] - df['Energy_res8'].mean())/df['Energy_res8'].std(ddof=0))
z9 = np.abs((df['Energy_res9'] - df['Energy_res9'].mean())/df['Energy_res9'].std(ddof=0))
z10 = np.abs((df['Energy_res10'] - df['Energy_res10'].mean())/df['Energy_res10'].std(ddof=0))

#Set the threshold for each residential and remove outliers according to zscore
threshold = 6.5  #6.5 sigma (example)

#create dfzi where all values of energy in residential i with z>sigma are removed
dfz1 = df[(z1 < 6.5)]
dfz2 = df[(z2 < 6.5)]
dfz3 = df[(z3 < 6.5)]
dfz4 = df[(z4 < 6.5)]
dfz5 = df[(z5 < 6.5)]
dfz6 = df[(z6 < 6.5)]
dfz7 = df[(z7 < 6.5)]
dfz8 = df[(z8 < 6.5)]
dfz9 = df[(z9 < 6.5)]
dfz10 = df[(z10 < 6.5)]
#after sorting the data we can notice that some of the upper values of energy consumption data was removed but probably these are real values and not outliers, while zeros where not rewmoved
# this is justified because seeing the histogram of residential 1 energy consumption data we see that it is not at all normally distributed

dfz1opp = df[(z1 > 6.5)]
dfz2opp = df[(z2 > 6.5)]
dfz3opp = df[(z3 > 6.5)]
dfz4opp = df[(z4 > 6.5)]
dfz5opp = df[(z5 > 6.5)]
dfz6opp = df[(z6 > 6.5)]
dfz7opp = df[(z7 > 6.5)]
dfz8opp = df[(z8 > 6.5)]
dfz9opp = df[(z9 > 6.5)]
dfz10opp = df[(z10 > 6.5)]


## IQR Method:
Q1_1 = df['Energy_res1'].quantile(0.25)
Q3_1 = df['Energy_res1'].quantile(0.75)
IQR_1 = Q3_1 - Q1_1

Q1_2 = df['Energy_res2'].quantile(0.25)
Q3_2 = df['Energy_res2'].quantile(0.75)
IQR_2 = Q3_2 - Q1_2

Q1_3 = df['Energy_res3'].quantile(0.25)
Q3_3 = df['Energy_res3'].quantile(0.75)
IQR_3 = Q3_3 - Q1_3

Q1_4 = df['Energy_res4'].quantile(0.25)
Q3_4 = df['Energy_res4'].quantile(0.75)
IQR_4 = Q3_4 - Q1_4

Q1_5 = df['Energy_res5'].quantile(0.25)
Q3_5 = df['Energy_res5'].quantile(0.75)
IQR_5 = Q3_5 - Q1_5

Q1_6 = df['Energy_res6'].quantile(0.25)
Q3_6 = df['Energy_res6'].quantile(0.75)
IQR_6 = Q3_6 - Q1_6

Q1_7 = df['Energy_res7'].quantile(0.25)
Q3_7 = df['Energy_res7'].quantile(0.75)
IQR_7 = Q3_7 - Q1_7

Q1_8 = df['Energy_res8'].quantile(0.25)
Q3_8 = df['Energy_res8'].quantile(0.75)
IQR_8 = Q3_8 - Q1_8

Q1_9 = df['Energy_res9'].quantile(0.25)
Q3_9 = df['Energy_res9'].quantile(0.75)
IQR_9 = Q3_9 - Q1_9

Q1_10 = df['Energy_res10'].quantile(0.25)
Q3_10 = df['Energy_res10'].quantile(0.75)
IQR_10 = Q3_10 - Q1_10

df_IQR_1 = df[((df['Energy_res1'] > (Q1_1 - 1.5 * IQR_1)) & (df['Energy_res1'] < (Q3_1 + 1.5 * IQR_1)))]
df_IQR_2 = df[((df['Energy_res2'] > (Q1_2 - 1.5 * IQR_2)) & (df['Energy_res2'] < (Q3_2 + 1.5 * IQR_2)))]
df_IQR_3 = df[((df['Energy_res3'] > (Q1_3 - 1.5 * IQR_3)) & (df['Energy_res3'] < (Q3_3 + 1.5 * IQR_3)))]
df_IQR_4 = df[((df['Energy_res4'] > (Q1_4 - 1.5 * IQR_4)) & (df['Energy_res4'] < (Q3_4 + 1.5 * IQR_4)))]
df_IQR_5 = df[((df['Energy_res5'] > (Q1_5 - 1.5 * IQR_5)) & (df['Energy_res5'] < (Q3_5 + 1.5 * IQR_5)))]
df_IQR_6 = df[((df['Energy_res6'] > (Q1_6 - 1.5 * IQR_6)) & (df['Energy_res6'] < (Q3_6 + 1.5 * IQR_6)))]
df_IQR_7 = df[((df['Energy_res7'] > (Q1_7 - 1.5 * IQR_7)) & (df['Energy_res7'] < (Q3_7 + 1.5 * IQR_7)))]
df_IQR_8 = df[((df['Energy_res8'] > (Q1_8 - 1.5 * IQR_8)) & (df['Energy_res8'] < (Q3_8 + 1.5 * IQR_8)))]
df_IQR_9 = df[((df['Energy_res9'] > (Q1_9 - 1.5 * IQR_9)) & (df['Energy_res9'] < (Q3_9 + 1.5 * IQR_9)))]
df_IQR_10 = df[((df['Energy_res10'] > (Q1_10 - 1.5 * IQR_10)) & (df['Energy_res10'] < (Q3_10 + 1.5 * IQR_10)))]


df_IQR1_opp = df[((df['Energy_res1'] < (Q1_1 - 1.5 * IQR_1)) | (df['Energy_res1'] > (Q3_1 + 1.5 * IQR_1)))]
df_IQR2_opp = df[((df['Energy_res2'] < (Q1_2 - 1.5 * IQR_2)) | (df['Energy_res2'] > (Q3_2 + 1.5 * IQR_2)))]
df_IQR3_opp = df[((df['Energy_res3'] < (Q1_3 - 1.5 * IQR_3)) | (df['Energy_res3'] > (Q3_3 + 1.5 * IQR_3)))]
df_IQR4_opp = df[((df['Energy_res4'] < (Q1_4 - 1.5 * IQR_4)) | (df['Energy_res4'] > (Q3_4 + 1.5 * IQR_4)))]
df_IQR5_opp = df[((df['Energy_res5'] < (Q1_5 - 1.5 * IQR_5)) | (df['Energy_res5'] > (Q3_5 + 1.5 * IQR_5)))]
df_IQR6_opp = df[((df['Energy_res6'] < (Q1_6 - 1.5 * IQR_6)) | (df['Energy_res6'] > (Q3_6 + 1.5 * IQR_6)))]
df_IQR7_opp = df[((df['Energy_res7'] < (Q1_7 - 1.5 * IQR_7)) | (df['Energy_res7'] > (Q3_7 + 1.5 * IQR_7)))]
df_IQR8_opp = df[((df['Energy_res8'] < (Q1_8 - 1.5 * IQR_8)) | (df['Energy_res8'] > (Q3_8 + 1.5 * IQR_8)))]
df_IQR9_opp = df[((df['Energy_res9'] < (Q1_9 - 1.5 * IQR_9)) | (df['Energy_res9'] > (Q3_9 + 1.5 * IQR_9)))]
df_IQR10_opp = df[((df['Energy_res10'] < (Q1_10 - 1.5 * IQR_10)) | (df['Energy_res10'] > (Q3_10 + 1.5 * IQR_10)))]



## Final decision to clean data from outliers:
#final decision made was to delete all points below a specific quantile for each residential, while there seems to be no high value outliers as mentioned before

#manually remove the only 2 zero values from residential 1 energy data:
df1 = df.drop(columns=['Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])
df1 = df1[df1['Energy_res1'] != 0]

#residential 2 energy consumption data has no outliers
df2 = df.drop(columns=['Energy_res1', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])

df3 = df.drop(columns=['Energy_res1', 'Energy_res2', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])
df3 = df3[df3['Energy_res3'] > df3['Energy_res3'].quantile(0.02)]

df4 = df.drop(columns=['Energy_res1', 'Energy_res2', 'Energy_res3', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])
df4 = df4[df4['Energy_res4'] != 0]

#manually remove the only 2 zero values from residential 5 energy data:
df5 = df.drop(columns=['Energy_res1', 'Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])
df5 = df5[df5['Energy_res5'] != 0]

#residential 6 energy consumption data has no outliers
df6 = df.drop(columns=['Energy_res1', 'Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res7', 'Energy_res8', 'Energy_res9', 'Energy_res10'])

df7 = df.drop(columns=['Energy_res1', 'Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res8', 'Energy_res9', 'Energy_res10'])
df7 = df7[df7['Energy_res7'] > df7['Energy_res7'].quantile(0.02)]

df8 = df.drop(columns=['Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res1', 'Energy_res9', 'Energy_res10'])
df8 = df8[df8['Energy_res8'] != 0]

df9 = df.drop(columns=['Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res1', 'Energy_res10'])
df9 = df9[df9['Energy_res9'] > df9['Energy_res9'].quantile(0.02)]

#manually remove the only 2 zero values from residential 10 energy data:
df10 = df.drop(columns=['Energy_res1', 'Energy_res2', 'Energy_res3', 'Energy_res4', 'Energy_res5', 'Energy_res6', 'Energy_res7', 'Energy_res8', 'Energy_res9'])   
df10 = df10[df10['Energy_res10'] != 0]

#Save each residential's energy data in a separate file as they will be used separately in the following steps
df1.to_csv('Clean_data/res1.csv', encoding='utf-8', index=True)
df2.to_csv('Clean_data/res2.csv', encoding='utf-8', index=True)
df3.to_csv('Clean_data/res3.csv', encoding='utf-8', index=True)
df4.to_csv('Clean_data/res4.csv', encoding='utf-8', index=True)
df5.to_csv('Clean_data/res5.csv', encoding='utf-8', index=True)
df6.to_csv('Clean_data/res6.csv', encoding='utf-8', index=True)
df7.to_csv('Clean_data/res7.csv', encoding='utf-8', index=True)
df8.to_csv('Clean_data/res8.csv', encoding='utf-8', index=True)
df9.to_csv('Clean_data/res9.csv', encoding='utf-8', index=True)
df10.to_csv('Clean_data/res10.csv', encoding='utf-8', index=True)

#save the same above data but without the meteo data (only for dashboard use in EDA tab)
df11 = df1.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df21 = df2.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df31 = df3.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df41 = df4.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df51 = df5.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df61 = df6.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df71 = df7.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df81 = df8.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df91 = df9.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])
df101 = df10.drop(columns=['Day of Week', 'AirTemp_C',
       'GlobalSolarRad_W/m2', 'PrecipitableWater_kg/m2', 'RelativeHumidity',
       'SnowDepth_LWE_cm', 'SurfacePressure_hPa', 'WindDirection10m',
       'WindSpeed10m_m/s', 'Hour'])

#these are the final eda results
df11.to_csv('Dashboard/res11.csv', encoding='utf-8', index=True)
df21.to_csv('Dashboard/res21.csv', encoding='utf-8', index=True)
df31.to_csv('Dashboard/res31.csv', encoding='utf-8', index=True)
df41.to_csv('Dashboard/res41.csv', encoding='utf-8', index=True)
df51.to_csv('Dashboard/res51.csv', encoding='utf-8', index=True)
df61.to_csv('Dashboard/res61.csv', encoding='utf-8', index=True)
df71.to_csv('Dashboard/res71.csv', encoding='utf-8', index=True)
df81.to_csv('Dashboard/res81.csv', encoding='utf-8', index=True)
df91.to_csv('Dashboard/res91.csv', encoding='utf-8', index=True)
df101.to_csv('Dashboard/res101.csv', encoding='utf-8', index=True)


#save the zscore data for the dashboard
dfz1 = dfz1[['Energy_res1']]
dfz2 = dfz2[['Energy_res2']]
dfz3 = dfz3[['Energy_res3']]


dfz1.to_csv('Dashboard/EDA_zscore1.csv', encoding='utf-8', index=True)
dfz2.to_csv('Dashboard/EDA_zscore2.csv', encoding='utf-8', index=True)
dfz3.to_csv('Dashboard/EDA_zscore3.csv', encoding='utf-8', index=True)

dfz1opp.to_csv('Dashboard/EDA_zscore1_opp.csv', encoding='utf-8', index=True)
dfz2opp.to_csv('Dashboard/EDA_zscore2_opp.csv', encoding='utf-8', index=True)
dfz3opp.to_csv('Dashboard/EDA_zscore3_opp.csv', encoding='utf-8', index=True)


#save IQR data for dashboard
df_IQR_4 = df_IQR_4[['Energy_res4']]
df_IQR_5 = df_IQR_5[['Energy_res5']]
df_IQR_6 = df_IQR_6[['Energy_res6']]


df_IQR_4.to_csv('Dashboard/EDA_IQR4.csv', encoding='utf-8', index=True)
df_IQR_5.to_csv('Dashboard/EDA_IQR5.csv', encoding='utf-8', index=True)
df_IQR_6.to_csv('Dashboard/EDA_IQR6.csv', encoding='utf-8', index=True)


df_IQR4_opp.to_csv('Dashboard/EDA_IQR4_opp.csv', encoding='utf-8', index=True)
df_IQR5_opp.to_csv('Dashboard/EDA_IQR5_opp.csv', encoding='utf-8', index=True)
df_IQR6_opp.to_csv('Dashboard/EDA_IQR6_opp.csv', encoding='utf-8', index=True)
