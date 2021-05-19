# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:37:14 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
import math

data_path = 'D:\\czy\\DecisionTree_stock\\StockData\\'

def label_threshold(sorted_arr,threshold): #给定阈值将因子分为0,1
    labeled_arr = np.where(sorted_arr <= threshold, 0, sorted_arr)
    lebeled_arr = np.where(sorted_arr > threshold, 1, labeled_arr)
    return lebeled_arr

#先把每个策略跑一遍，处理好，得到一个factor_dict
factor_dict = {1:factordata}

#通过训练集得到一个final_tree
decision_path_list = final_tree['decision_path']

data_shape = ret_clsfd.shape #ret_clsfd或者别的随便什么

position_matrix = np.zeros(data_shape) 
for i in range(len(decision_path_list)): #对于每一路径
    tmp = np.full(data_shape, i)
    for j in decision_path_list[i]: #对于路径上每一节点
        strategy_name, threshold, zero_or_one = j[0], j[1], j[2]
        thresed_factor = label_threshold(factor_dict[strategy_name],threshold)
        tmp = np.where(thresed_factor==j[2],tmp,0)
    position_matrix += tmp