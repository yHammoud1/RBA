# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:39:02 2021

@author: Yara
"""

import numpy as np
import pandas as pd
import datetime
from functools import reduce

########### Import Data ################################################################
## Each res file contains the energy consumption (kWh) data of one of the residential buildings
res1 = pd.read_csv('./dataverse_files/Residential_3.csv', dayfirst=True);
res2 = pd.read_csv('./dataverse_files/Residential_4.csv', dayfirst=True);
res3 = pd.read_csv('./dataverse_files/Residential_5.csv', dayfirst=True);
res4 = pd.read_csv('./dataverse_files/Residential_6.csv', dayfirst=True);
res5 = pd.read_csv('./dataverse_files/Residential_8.csv', dayfirst=True);
res6 = pd.read_csv('./dataverse_files/Residential_10.csv', dayfirst=True);
res7 = pd.read_csv('./dataverse_files/Residential_11.csv', dayfirst=True);
res8 = pd.read_csv('./dataverse_files/Residential_12.csv', dayfirst=True);
res9 = pd.read_csv('./dataverse_files/Residential_13.csv', dayfirst=True);
res10 = pd.read_csv('./dataverse_files/Residential_14.csv', dayfirst=True);

## The buildings are in the same region so meteo data from one station apply to all buildings
meteo = pd.read_csv('49.282729_-123.120738_Solcast_PT60M.csv', index_col='PeriodStart', parse_dates=['PeriodStart']);

holidays = pd.read_csv('Holidays.csv', parse_dates=['date'], index_col='date', dayfirst=True);



########### Data Cleaning ################################################################

## Cleaning of meteo data
meteo.index = meteo.index.strftime('%Y-%m-%d %H:%M:%S');
meteo.index = pd.to_datetime(meteo.index);
meteo = meteo.drop(columns=['PeriodEnd', 'Period'], axis=1)

## Cleaning of residential buildings energy consumption data
def fixdates(res):
    res['hour'] = [datetime.time(num).strftime("%H:00:00") for num in res['hour']];
    res['DateTime'] = res['date'].astype(str) + ' ' + res['hour'];
    res['DateTime'] = pd.to_datetime(res['DateTime'],format="%Y/%m/%d %H:%M:%S");
    res = res.set_index('DateTime', drop=True);
    res = res.drop(columns=['date', 'hour'], axis=1); 
    return res


res1 = fixdates(res1)
res2 = fixdates(res2)
res3 = fixdates(res3)
res4 = fixdates(res4)
res5 = fixdates(res5)
res6 = fixdates(res6)
res7 = fixdates(res7)
res8 = fixdates(res8)
res9 = fixdates(res9)
res10 = fixdates(res10)

res1 = res1.rename(columns={'energy_kWh':'Energy_res1'})
res2 = res2.rename(columns={'energy_kWh':'Energy_res2'})
res3 = res3.rename(columns={'energy_kWh':'Energy_res3'})
res4 = res4.rename(columns={'energy_kWh':'Energy_res4'})
res5 = res5.rename(columns={'energy_kWh':'Energy_res5'})
res6 = res6.rename(columns={'energy_kWh':'Energy_res6'})
res7 = res7.rename(columns={'energy_kWh':'Energy_res7'})
res8 = res8.rename(columns={'energy_kWh':'Energy_res8'})
res9 = res9.rename(columns={'energy_kWh':'Energy_res9'})
res10 = res10.rename(columns={'energy_kWh':'Energy_res10'})

## Merging the energy consumption data for all buildings together
data_frames = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10]

energy_data = reduce(lambda  left,right: pd.merge(left,right,on=['DateTime'],
                                            how='outer'), data_frames)


## Croping the data so that the energy data used for all buildings have same start and end date
# with the final duration of available data being from 21-02-2015 to 29-01-2018
energy_data = energy_data[energy_data.index >= '2015-02-21 00:00:00' ]
energy_data = energy_data[energy_data.index < '2018-01-30 00:00:00' ]

## Cleaning of holiday data
holidays['weekend'] = np.where(pd.isnull(holidays['holiday']), holidays.index.dayofweek, holidays['weekend'])
holidays['weekend'] = np.where(pd.isnull(holidays['holiday']), holidays['weekend'],7)
holidays = holidays.drop(columns=['dst', 'holiday'], axis=1)
holidays = holidays.rename(columns={'weekend':'Day Type'})


## Merge holiday data with energy data
energy_data['date only'] = [d.date() for d in energy_data.index]
energy_data['date only'] = pd.to_datetime(energy_data['date only'], format='%Y-%m-%d')

energy_data = energy_data.reset_index(drop=False)
holidays = holidays.reset_index(drop=False)

all_data = pd.merge(energy_data, holidays, left_on='date only', right_on='date', how='outer')
all_data = all_data.set_index('DateTime', drop = True)
all_data = all_data.drop(columns=['date only', 'date', 'day'], axis=1)


## Merge all energy data (and holidays) with meteo data
all_data = pd.merge(all_data, meteo, left_on=all_data.index, right_on=meteo.index, how='outer')
all_data = all_data.rename(columns={'key_0':'Date', 'AirTemp':'AirTemp_C', 
                           'Ghi':'GlobalSolarRad_W/m2', 
                           'PrecipitableWater':'PrecipitableWater_kg/m2',
                           'SnowWater':'SnowDepth_LWE_cm',
                           'SurfacePressure':'SurfacePressure_hPa', 
                           'WindSpeed10m':'WindSpeed10m_m/s'})

## Save final dataframe
all_data.to_csv('Proj3_clean_data.csv', encoding='utf-8', index=False)
