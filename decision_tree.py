# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:15:08 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
import math

data_path = 'D:\\czy\\DecisionTree_stock\\StockData\\'
root_path = 'D:\\czy\\DecisionTree_stock\\'
ret_clsfd = pd.read_csv(data_path + 'ret_clsfd.csv').set_index('DATETIME')
ret_clsfd = ret_clsfd.shift(-2)
#shift

def cal_entropy(df):
    arr = np.array(df)
    count1, count0 = np.sum(arr==1), np.sum(arr==0)
    total = count0 + count1
    ratio1, ratio0 = count1/total, count0/total
    entropy = - ratio1 * math.log2(ratio1) - ratio0 * math.log2(ratio0)
    return entropy, total

def standardize_factor_df(factor_df): #标准化到01
    arr = np.array(factor_df)
    sorted_arr = arr.argsort().argsort()
    sorted_arr = np.where(np.isnan(arr),np.nan,sorted_arr)
    max_arr = np.nan_to_num(sorted_arr).max(1)
    sorted_arr = sorted_arr / max_arr.reshape(max_arr.shape[0],1)
    return sorted_arr

def label_threshold(sorted_arr,threshold): #给定阈值将因子分为0,1
    labeled_arr = np.where(sorted_arr <= threshold, 0, sorted_arr)
    lebeled_arr = np.where(sorted_arr > threshold, 1, labeled_arr)
    return lebeled_arr

def set_min_entro(entro_list, dispersable_sign_list): #设置熵最小的节点的索引
    min_index = 0
    entro = 10
    for i in range(len(entro_list)):
        if dispersable_sign_list[i] == True and entro_list[i] <= entro:
            entro = entro_list[i]
            min_index = i
    return min_index

def make_decision(factor_dict, root_data, init_entro, old_path):
    threshold_list = [0.3,0.4,0.5,0.6,0.7]
    strategy_name_list = [1,2,3]
    min_entro = init_entro
    using_strategy,using_thres = 10,10 #记录目前最优因子
    data0,data1 = 10,10
    signal = False #标志可分或不可分
    for strategy_name in strategy_name_list:
        for thres in threshold_list:
            print([strategy_name, thres])
            if ([strategy_name, thres, 1]  not in old_path) and ([strategy_name, thres, 0] not in old_path): #该因子是否被用过
                thresed_factor = label_threshold(factor_dict[strategy_name],thres)
                tmp_data1 = np.where(thresed_factor==1, root_data, np.nan)
                tmp_data0 = np.where(thresed_factor==0, root_data, np.nan)
                if  np.count_nonzero(~np.isnan(tmp_data0)) > 2000 and np.count_nonzero(~np.isnan(tmp_data1)) > 2000:
                    e1,t1 = cal_entropy(tmp_data1)
                    e0,t0 = cal_entropy(tmp_data0)
                    new_entro = (e1 * t1 + e0 * t0) / (t1 + t0)
                    if new_entro < min_entro: #该分类是否对于原节点的熵有增益
                        min_entro = new_entro
                        signal = True
                        using_strategy,using_thres = strategy_name, thres
                        data1 = tmp_data1
                        data0 = tmp_data0        
    
    if signal:
        print('节点初始熵：'+ str(init_entro) + ', 最优熵：'+str(min_entro), end=', ')
    return signal, [data0,data1], [using_strategy, using_thres]    #先0再1
             
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












            
factor_dict = {} #array, not DataFrame            
for i in range(84):
    factor_dict[i] = np.array(pd.read_csv(root_path + 'factors_value\\%d.csv'%i,index_col=0))
            
tree = [ret_clsfd]
decision_path_list = [[]]
i = 0
entro_list = [cal_entropy(tree[0])[0]]
count_list = [[0,0]] #记录每个节点内部个数，先0再1
dispersable_sign_list = [True]

while len(tree) <= 2000 and i < len(tree) and (True in dispersable_sign_list):
    root_data = tree[i]
    init_entro = entro_list[i]
    signal, [data0,data1], [using_strategy, using_thres] = make_decision(factor_dict, root_data, init_entro, decision_path_list[i])
    if signal: #如果可分
        print('节点%d已分'%i)
        path0, path1 = decision_path_list[i]+[[using_strategy, using_thres,0]], decision_path_list[i]+[[using_strategy, using_thres,1]]
        decision_path_list += [path0, path1]
        del decision_path_list[i]
        tree += [data0,data1]
        del tree[i]
        dispersable_sign_list += [True,True]
        del dispersable_sign_list[i]
        entro_list += [cal_entropy(data0)[0], cal_entropy(data1)[0]]
        del entro_list[i]
        count_list += [[np.sum(data0==0), np.sum(data0==1)], [np.sum(data1==0), np.sum(data1==1)]]
        del count_list[i]
        i = set_min_entro(entro_list,dispersable_sign_list) #设置为下一个可分的最小熵节点
        print('转至节点%d'%i)
    else: #如果不可分
        dispersable_sign_list[i] = False
        print('节点%d不可分'%i)
        if True in dispersable_sign_list:
            i = set_min_entro(entro_list,dispersable_sign_list) #设置为下一个可分的最小熵节点
            print('转至节点%d'%i)
        else: #剩下的全部都不可分了
            break

ratio_of1_list = [(i[1]/(i[1]+i[0])) for i in count_list] #每个子节点的含1率
final_tree = pd.DataFrame({'decision_path':decision_path_list,'entro':entro_list,'count0and1':count_list,'1ratio':ratio_of1_list}).sort_values(by='1ratio',ascending = False)                 

argsort_arr = np.argsort(np.array(ratio_of1_list)) #每个子节点含1率的排名
position_matrix = np.zeros(ret_clsfd.shape) 
for i in range(len(tree)):
    position_matrix += np.where(~np.isnan(tree[i]),argsort_arr[i],0)
    










