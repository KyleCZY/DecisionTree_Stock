# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:42:51 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
import math
import dill
from datetime import datetime
import os

data_path = 'D:\\czy\\DecisionTree_stock\\StockData\\'
root_path = 'D:\\czy\\DecisionTree_stock\\'


raw_ret = pd.read_csv(data_path + 'RawRet.csv').set_index('DATETIME')
zzret = pd.read_csv(data_path + 'zz500dailyret.csv').set_index('Date')
new = raw_ret.join(zzret)['Ret']
ret = raw_ret.apply(lambda x: x - new)

ret3 = ret.rolling(3).sum().shift(-4)
classified = ret3.apply(lambda x: np.sign(abs(x) + x))
classified.to_csv(data_path + 'ret_clsfd3.csv')