# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:56:50 2021

@author: Yara
"""
import numpy as np
import pandas as pd
import matplotlib as plt

df = pd.read_csv('Proj3_clean_data.csv', dayfirst=False)
df = df.set_index('Date', drop=True)

########### Display Data ################################################################
plt.plot(df['Energy_res1'])