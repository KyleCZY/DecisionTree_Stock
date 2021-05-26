# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:37:14 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
import math

data_path = 'D:\\czy\\DecisionTree_stock\\StockData\\'
root_path = 'D:\\czy\\DecisionTree_stock\\'

yizi = pd.read_csv(data_path + 'yizi.csv').set_index('DATETIME')[2431:] 
RawRet = pd.read_csv(data_path + 'RawRet.csv').set_index('DATETIME')[2431:]
yizi = yizi.applymap(lambda x: round(x, 2))
RawRet = RawRet.applymap(lambda x: round(x, 2))
index_for_all = RawRet.index
columns_for_all = RawRet.columns
yizi = np.array(yizi)
RawRet = np.array(RawRet)

def label_threshold(sorted_arr,threshold): #给定阈值将因子分为0,1
    labeled_arr = np.where(sorted_arr <= threshold, 0, sorted_arr)
    lebeled_arr = np.where(sorted_arr > threshold, 1, labeled_arr)
    return lebeled_arr

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
    
    perform = pd.DataFrame()
    perform.loc[0,'年化收益率'] = Retdf['Ret'].mean() * 243
    perform.loc[0,'年化波动率'] = Retdf['Ret'].std() * np.sqrt(243)
    perform.loc[0,'日胜率'] = np.sum(Retdf['Ret']>0)/len(Retdf)
    perform.loc[0,'日盈亏比'] = -Retdf.loc[Retdf['Ret']>0,'Ret'].mean()/Retdf.loc[Retdf['Ret']<0,'Ret'].mean()
    
    perform.loc[0,'最大回撤'] = Retdf['dd'].min()
    perform.loc[0,'夏普'] = perform.loc[0,'年化收益率']/perform.loc[0,'年化波动率']
    perform.loc[0,'卡玛'] = -perform.loc[0,'年化收益率']/perform.loc[0,'最大回撤']
    perform.loc[0,'日均换手率'] = factor.diff().abs().sum(axis = 1).mean()/2/stock_num
    
    return calma, Retdf, perform

#先把每个策略跑一遍，处理好，得到一个factor_dict
factor_dict = {} #array, not DataFrame            
for i in range(84):
    factor_dict[i] = np.array(pd.read_csv(root_path + 'factors_value\\%d.csv'%i,index_col=0)[2431:])
    print('\rloading factor%d'%i,end = ' ') 

#通过训练集得到一个final_tree
decision_path_list = final_tree['decision_path']

data_shape = yizi.shape #或者别的随便什么
position_matrix = np.zeros(data_shape) 
for i in range(len(decision_path_list)): #对于每一路径
    tmp = np.full(data_shape, i)
    for j in decision_path_list[i]: #对于路径上每一节点
        strategy_name, threshold, zero_or_one = j[0], j[1], j[2]
        thresed_factor = label_threshold(factor_dict[strategy_name],threshold)
        tmp = np.where(thresed_factor==j[2],tmp,0)
    position_matrix -= tmp

calma, Retdf, perform = fitness_func(position_matrix,yizi,RawRet)