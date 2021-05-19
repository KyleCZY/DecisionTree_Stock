# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:56:24 2021

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


#%%回测函数               
def fitness_func(factor_tmp, yizi,RawRet):
    #计算持仓
    stock_num = 300
    fee_oneside = 0.0007
    factor = factor_tmp.argsort(axis = 1).argsort(axis = 1)
    factor = factor + np.isnan(factor_tmp)*5000
    factor = (factor<stock_num)
    factor = factor.astype("int")
    factor = factor[:-1,:]

    #涨跌停或停牌持仓调控
    iftrade = yizi.copy()
    iftrade[iftrade == 1] = np.nan
    iftrade[iftrade == 0] = 1
    factor_array = np.multiply(factor,np.array(iftrade)[1:,:])
    factor_array = factor_array[:-1,:]
    factor = pd.DataFrame(factor_array, index = index_for_all[2:], columns = columns_for_all)
    factor = factor.fillna(method = 'pad').fillna(0)

    #清算所选股票收益和换手
    retmat = RawRet.copy()
    retposmat = np.multiply(np.array(factor),retmat[2:])
    retposmat = pd.DataFrame(retposmat, index = index_for_all[2:], columns = columns_for_all)
    fee = factor.diff().abs()*fee_oneside
    retposmat = retposmat - fee
    Retdf = retposmat.sum(axis = 1)/stock_num
    Retdf = Retdf.reset_index().rename(columns = {0:'Ret'})
        
    #计算各项指标
    Retdf['Retcumsum'] = Retdf['Ret'].cumsum()
    Retdf['Retcumsummax'] = Retdf['Retcumsum'].cummax() 
    Retdf['dd'] = Retdf['Retcumsum'] - Retdf['Retcumsummax'] 
    calma = -Retdf['Ret'].sum()/Retdf['dd'].min()
    '''
    perform = pd.DataFrame()
    perform.loc[0,'年化收益率'] = Retdf['Ret'].mean() * 243
    perform.loc[0,'年化波动率'] = Retdf['Ret'].std() * np.sqrt(243)
    perform.loc[0,'日胜率'] = np.sum(Retdf['Ret']>0)/len(Retdf)
    perform.loc[0,'日盈亏比'] = -Retdf.loc[Retdf['Ret']>0,'Ret'].mean()/Retdf.loc[Retdf['Ret']<0,'Ret'].mean()
    
    perform.loc[0,'最大回撤'] = Retdf['dd'].min()
    perform.loc[0,'夏普'] = perform.loc[0,'年化收益率']/perform.loc[0,'年化波动率']
    perform.loc[0,'卡玛'] = -perform.loc[0,'年化收益率']/perform.loc[0,'最大回撤']
    perform.loc[0,'日均换手率'] = factor.diff().abs().sum(axis = 1).mean()/2/stock_num
    '''
    return calma, Retdf

#%%回测单因子多参数
factorNo = 29
f = getattr(optr, 'factor' + str(factorNo))

#设置参数，不同因子要修改
A_list = ['close','close_adj','high_adj','low_adj','open_adj','volume_adj','vwap_adj']
B_list = ['close','close_adj','high_adj','low_adj','open_adj','volume_adj','vwap_adj']
C_list = ['close','close_adj','high_adj','low_adj','open_adj','volume_adj','vwap_adj']
D_list = ['close','close_adj','high_adj','low_adj','open_adj','volume_adj','vwap_adj']
a_list = [15,30,45]
b_list = [15,30,45]
c_list = [15,30,45]
d_list = [15,30,45]
e_list = [15,30,45]
paraLists =[]
paraLists.append(A_list)
paraLists.append(B_list)
paraLists.append(C_list)
#paraLists.append(D_list)
paraLists.append(a_list)
paraLists.append(b_list)
#paraLists.append(c_list)
#paraLists.append(d_list)
#paraLists.append(e_list)
paraRows = list(itertools.product(*paraLists))
#paraRows_select = paraRows


paraRows_select = []
for p in paraRows:
    if p[1] != p[2]: #删除某些参数组，条件自定
        paraRows_select.append(p)
'''
'''
#测试因子
calma_result = pd.DataFrame()
for i,p in enumerate(paraRows_select):

    factor_tmp = f(Dataslide[p[0]],Dataslide[p[1]],Dataslide[p[2]],p[3],p[4]) #不同因子要修改

    [calma, Retdf] = fitness_func(factor_tmp, yizi,RawRet)
    calma_result.loc[i, 'calma'] = calma
    calma_result.loc[i, 'para'] = str(p)
    print ("\r进度：" + str(round(i/len(paraRows_select)*100, 2)) + "%", end = ' ') 
    
calma_result.to_csv(outpath + "calma_factor" + str(factorNo) + ".csv")






