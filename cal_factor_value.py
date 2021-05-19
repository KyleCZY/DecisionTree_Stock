# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:23:48 2021

@author: Administrator
"""

import random
import itertools
import pandas as pd
import numpy as np
import os
rootpath = "D:\\czy\\DecisionTree_stock\\" #存放optr的地址
os.chdir(rootpath)
import optr
import datetime
import importlib
importlib.reload(optr)
path = 'D:\czy\DecisionTree_stock\\StockData\\'#数据地址
outpath = "D:\\czy\\DecisionTree_stock\\output\\"

print ("setting data...")
time_interval = 20 #default_time_interval在optr.py中调整
optr.change_time_interval(time_interval)
data_list = ['close','close_adj','high_adj','low_adj','open_adj','volume_adj','vwap_adj']
yizi = pd.read_csv(path + 'yizi.csv').set_index('DATETIME') 
RawRet = pd.read_csv(path + 'RawRet.csv').set_index('DATETIME') 
yizi = yizi.applymap(lambda x: round(x, 2))
RawRet = RawRet.applymap(lambda x: round(x, 2))
index_for_all = RawRet.index
columns_for_all = RawRet.columns
yizi = np.array(yizi)
RawRet = np.array(RawRet)
Dataslide = {}
for tag in data_list:
    Dataslide[tag] = pd.read_csv(path + tag + '.csv').set_index('DATETIME')
    Dataslide[tag] = Dataslide[tag].fillna(method = 'pad')
    Dataslide[tag] = Dataslide[tag].applymap(lambda x: round(x, 2))
    Dataslide[tag] = np.array(Dataslide[tag])
print ("data settled")


#%%
factors = np.array(pd.read_csv(rootpath+'select_factors.csv'))

def standardize_factor_df(factor_df): #标准化到01
    arr = np.array(factor_df)
    sorted_arr = arr.argsort().argsort()
    sorted_arr = np.where(np.isnan(arr),np.nan,sorted_arr)
    max_arr = np.nan_to_num(sorted_arr).max(1)
    sorted_arr = sorted_arr / max_arr.reshape(max_arr.shape[0],1)
    return sorted_arr


for i in range(len(factors)):
    data = eval('optr.factor' + (factors[i][0]))
    data = standardize_factor_df(data)
    pd.DataFrame(data).to_csv(rootpath+'factors_value\\%d.csv'%i)
    print('\r%d'%i)