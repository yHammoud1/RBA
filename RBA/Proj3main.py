# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:39:02 2021

@author: Yara
"""

import pandas as pd
import datetime
from functools import reduce

########### 1. Import Data ################################################################
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

meteo = pd.read_csv('49.282729_-123.120738_Solcast_PT60M.csv', index_col='PeriodStart', parse_dates=['PeriodStart']);
holidays = pd.read_csv('Holidays.csv', parse_dates=['date'], index_col='date', dayfirst=True);

meteo.index = meteo.index.strftime('%Y-%m-%d %H:%M:%S');
meteo.index = pd.to_datetime(meteo.index);


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

data_frames = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10]


df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['DateTime'],
                                            how='outer'), data_frames)


# trial = pd.merge(res1, meteo, how='outer', left_on=res1.index , right_on=meteo.index)
# trial = trial.rename(columns={'key_0':'Date'})

