# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:34:25 2021

"""

import numpy as np
EPS_F64 = 1.e-12
#index为Date或Hours

default_time_interval = 10
    
def change_time_interval(x):
    global default_time_interval
    default_time_interval = x
    return None

#rolling函数
def mat_rolling(A, period):  # 针对2D arr, place rolling at z-axis quick than at x-axis
    B = np.concatenate([np.full((period-1,) + A.shape[1:], np.nan), A], axis=0)
    shape = A.shape + (period,)
    strides = B.strides + (B.strides[0],)
    return np.lib.stride_tricks.as_strided(B, shape=shape, strides=strides)


#加法
def add(a, b):
    return a + b

#减法
def sub(a, b):
    return a - b

#乘法
def mul(a, b):
    return a * b

#除法
def div(a, b): 
    return a / b

def sign(a):
    return np.sign(a)

#开方，有负数则不开
def sqrt(a):
    if np.any(a<0):
        return a
    else:
        return a ** 0.5

#取对数，有负数则不取
def log(a):
    if np.any(a<=0):
        return a
    else:
        return np.log(a)

#取绝对值
def abso(a):
    return np.abs(a)

#平方
def sqr(a):
    return a**2

#倒数，有0则不取
def inv(a):
    if np.any(a==0):
        return a
    else:
        return 1 / a

#最大值    
def maxm(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (a>b)*a+(b>a)*b
    else:
        return max(a, b)

#最小值 
def minm(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (a<b)*a+(b<a)*b
    else:
        return min(a, b)

#均值
def mean(a, b):
    return (a + b) / 2

#a在b个时间之前的值。两个数字则返回NaN，两个矩阵则返回第一个矩阵和10d的结果
def delay(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        return np.concatenate([np.full((default_time_interval-1,a.shape[1]), np.nan), a[:-default_time_interval+1,:]], axis=0)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b > 0.5:
                b=int(b)
                return np.concatenate([np.full((b,a.shape[1]), np.nan), a[:-b,:]], axis=0)
            else:
                return np.concatenate([np.full((default_time_interval,a.shape[1]), np.nan), a[:-default_time_interval,:]], axis=0)
        else:
            if a > 0.5:
                a=int(a)
                return np.concatenate([np.full((a,b.shape[1]), np.nan), b[:-a,:]], axis=0)
            else:
                return np.concatenate([np.full((default_time_interval,b.shape[1]), np.nan), b[:-default_time_interval,:]], axis=0)       
    else: #两个数字
        return -1

#a在b个时间长度上的差分，两个数字则返回NaN，两个矩阵则返回第一个矩阵和10d的结果
def delta_func(A, p):
    t1 = A[p-1:,:]-A[:-p+1,:]
    t2 = np.full([p-1, A.shape[1]], np.nan)   
    return np.concatenate((t2, t1),axis=0)    
    
def delta(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        return delta_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return delta_func(a, int(b))
            else:
                return delta_func(a, default_time_interval)
        else:
            if a >= 2:
                return delta_func(b, int(a))
            else:
                return delta_func(b, default_time_interval) 
    else: #两个数字
        return -1
 
       
#a在b个时间长度上的变化率，两个数字则返回NaN，两个矩阵则返回第一个矩阵和10d的结果  
def delta_perc_func(A, p):
    t1 = A[p-1:,:]/A[:-p+1,:]-1
    t2 = np.full([p-1, A.shape[1]], np.nan)   
    return np.concatenate((t2, t1),axis=0)

def delta_perc(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df
        if (~con_a) and (~con_b):
            return -1
        elif (~con_a):       
            return delta_perc_func(b, default_time_interval)
        else:
            return delta_perc_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:   
                b=int(b)
                return delta_perc_func(a, b)
            else:     
                return delta_perc_func(a, default_time_interval)
        else:
            if a >= 2:        
                a=int(a)
                return delta_perc_func(b, a)
            else:
                return delta_perc_func(b, default_time_interval)     
    else: #两个数字
        return -1

#负数
def neg(a):
    return 0 - a

#截面排序
def rank(a):
    if isinstance(a, np.ndarray):
        t = a.argsort(axis = 1).argsort(axis = 1).astype(float)
        t[np.isnan(a)] = np.nan
        return t
    else:
        return -1

#截面归一化
def scale(a):
    if isinstance(a, np.ndarray):
        a_mean = np.nanmean(a, axis = 1)
        a_std = np.nanstd(a, axis = 1)
        a_mean = np.repeat(a_mean, a.shape[1]).reshape(a.shape[0],a.shape[1])
        a_std = np.repeat(a_std, a.shape[1]).reshape(a.shape[0],a.shape[1])
        return (a - a_mean)/a_std
    else:
        return -1

#a过去b个时间内最大值的下标, rolling_window
def argmax_func(A, p):
    A_ = mat_rolling(A, p)
    t = A_.argmax(axis = 2)
    return t

def rw_argmax(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df
        if ~con_a and ~con_b:
            return -1
        elif ~con_a:        
            return argmax_func(b, default_time_interval)
        else:
            return argmax_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return argmax_func(a, int(b))
            else:
                return argmax_func(a, default_time_interval)
        else:
            if a >= 2:
                return argmax_func(b, int(a))
            else:
                return argmax_func(b, default_time_interval)
    else: #两个数字
        return -1

#a过去b个时间内最小值的下标
def argmin_func(A, p):
    A_ = mat_rolling(A, p)
    t = A_.argmax(axis = 2)
    return t

def rw_argmin(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df
        if ~con_a and ~con_b:
            return -1
        elif ~con_a:        
            return argmin_func(b, default_time_interval)
        else:
            return argmin_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return argmin_func(a, int(b))
            else:
                return argmin_func(a, default_time_interval)
        else:
            if a >= 2:
                return argmin_func(b, int(a))
            else:
                return argmin_func(b, default_time_interval)
    else: #两个数字
        return -1

#a过去b个时间内最大最小值的下标之差
def rw_argmaxmin(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df
        if ~con_a and ~con_b:
            return -1
        elif ~con_a:        
            return argmax_func(b, default_time_interval) - argmin_func(b, default_time_interval)
        else:
            return argmax_func(a, default_time_interval) - argmin_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                b=int(b)
                return argmax_func(a, b) - argmin_func(a, b)
            else:
                return argmax_func(a, default_time_interval) - argmin_func(a, default_time_interval)
        else:
            if a >= 2:
                a=int(a)
                return argmax_func(b, a) - argmin_func(b, a)
            else:
                return argmax_func(b, default_time_interval) - argmin_func(b, default_time_interval)
    else: #两个数字
        return -1


#a过去b个时间内最大值
def max_func(A, p):
    A_ = mat_rolling(A, p)
    t = np.nanmax(A_,axis = 2)
    return t

def rw_max(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df

        if ~con_a and ~con_b:
            return -1
        elif ~con_a:        
            return max_func(b, default_time_interval)
        else:
            return max_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return max_func(a, int(b))
            else:
                return max_func(a, default_time_interval)
        else:
            if a >= 2:
                return max_func(b, int(a))
            else:
                return max_func(b, default_time_interval)
    else: #两个数字
        return -1  
    
#a过去b个时间内最小值
def min_func(A, p):
    A_ = mat_rolling(A, p)
    t = np.nanmin(A_,axis = 2)
    return t

def rw_min(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df
        if ~con_a and ~con_b:
            return -1
        elif ~con_a:        
            return min_func(b, default_time_interval)
        else:
            return min_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return min_func(a, int(b))
            else:
                return min_func(a, default_time_interval)
        else:
            if a >= 2:
                return min_func(b, int(a))
            else:
                return min_func(b, default_time_interval)
    else: #两个数字
        return -1  


    
#a过去b个时间内均值
def ma_func(A, p):
    A_ = mat_rolling(A, p)
    t = np.nanmean(A_,axis = 2)
    return t

def rw_ma(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
        con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
        if ~con_a and ~con_b:
            return -1
        elif ~con_a:          
            return ma_func(b, default_time_interval)
        else:
            return ma_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return ma_func(a, int((b)))
            else:
                return ma_func(a, default_time_interval)
        else:
            if a >= 2:
                return ma_func(b, int(a))
            else:
                return ma_func(b, default_time_interval)
    else: #两个数字
        return -1  

#a过去b个时间内标准差
def std_func(A, p):
    A_ = mat_rolling(A, p)
    t = np.nanstd(A_,axis = 2)
    return t

def rw_std(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df
        if ~con_a and ~con_b:
            return -1
        elif ~con_a:          
            return std_func(b, default_time_interval)
        else:
            return std_func(a, default_time_interval)
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return std_func(a, int((b)))
            else:
                return std_func(a, default_time_interval)
        else:
            if a >= 2:
                return std_func(b, int(a))
            else:
                return std_func(b, default_time_interval)
    else: #两个数字
        return -1  

#maxmin标准化
def rw_maxmin_norm(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df
        if ~con_a and ~con_b:
            return -1
        elif ~con_a:        
            return (b - min_func(b,default_time_interval)) / (max_func(b,default_time_interval) - min_func(b,default_time_interval))
        else:
            return (a - min_func(a,default_time_interval)) / (max_func(a,default_time_interval) - min_func(a,default_time_interval))
    elif a_type or b_type: #1数字1df
        if a_type:
            if ~con_a:
                return a
            else:
                if b >= 2:
                    return (a - min_func(a,int(b))) / (max_func(a,int(b)) - min_func(a,int(b)))
                else:
                    return (a - min_func(a,default_time_interval)) / (max_func(a,default_time_interval) - min_func(a,default_time_interval))
        else:
            if ~con_b:
                return b
            else:
                if a >= 2:
                    return (b - min_func(b,int(a))) / (max_func(b,int(a)) - min_func(b,int(a)))
                else:
                    return (b - min_func(b,default_time_interval)) / (max_func(b,default_time_interval) - min_func(b,default_time_interval))  
    else: #两个数字
        return -1 

#a目前取值在过去b时间的排名
def rw_rank(a,b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
    con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
    if a_type and b_type: #两个df
        if ~con_a and ~con_b:
            return -1
        elif ~con_a:        
            return mat_rolling(b, default_time_interval).argsort(axis = -1).argsort(axis=-1)[:,:,-1]
        else:
            return mat_rolling(a, default_time_interval).argsort(axis = -1).argsort(axis=-1)[:,:,-1]
    elif a_type or b_type: #1数字1df
        if a_type:
            if b >= 2:
                return mat_rolling(a, int(b)).argsort(axis = -1).argsort(axis=-1)[:,:,-1]
            else:
                return mat_rolling(a, default_time_interval).argsort(axis = -1).argsort(axis=-1)[:,:,-1]
        else:
            if a >= 2:
                return mat_rolling(b, int(a)).argsort(axis = -1).argsort(axis=-1)[:,:,-1]
            else:
                return mat_rolling(b, default_time_interval).argsort(axis = -1).argsort(axis=-1)[:,:,-1]
    else: #两个数字
        return -1 
    
#相关系数
def rw_corr(a, b,time_interval):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
        con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
        if ~con_a or ~con_b:
            return -1
        else:
            x_mean = 1.0 * rw_ma(a, time_interval)
            y_mean = 1.0 * rw_ma(b, time_interval)
            xx_mean = rw_ma(a * a, time_interval)
            xy_mean = rw_ma(a * b, time_interval)
            yy_mean = rw_ma(b * b, time_interval)
        
            p = xy_mean - x_mean * y_mean
            q1 = xx_mean - x_mean * x_mean
            q2 = yy_mean - y_mean * y_mean
            q1[q1 <= 0.0] = EPS_F64
            q2[q2 <= 0.0] = EPS_F64
            return p / np.sqrt(q1 * q2)
    else:
        return -1
        
#回归的贝塔系数
def rw_beta(a, b, time_interval):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
        con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
        if ~con_a or ~con_b:
            return -1
        else:
            x_mean = 1.0 * rw_ma(a, time_interval)
            y_mean = 1.0 * rw_ma(b, time_interval)
            xx_mean = rw_ma(a * a, time_interval)
            xy_mean = rw_ma(a * b, time_interval)
            p = xy_mean - x_mean * y_mean
            q = xx_mean - x_mean * x_mean
            q[q <= 0.0] = EPS_F64
            return p / q
    else:
        return -1

#回归的alpha系数
def rw_alpha(a, b, time_interval):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
        con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
        if ~con_a or ~con_b:
            return -1
        else:
            x_mean = 1.0 * rw_ma(a, time_interval)
            y_mean = 1.0 * rw_ma(b, time_interval)
            xx_mean = rw_ma(a * a, time_interval)
            xy_mean = rw_ma(a * b, time_interval)
            p = xy_mean - x_mean * y_mean
            q = xx_mean - x_mean * x_mean
            q[q <= 0.0] = EPS_F64
            beta = p / q
            return y_mean - beta * x_mean
    else:
        return -1

#比较两个矩阵大小，返回01矩阵大
def mat_big(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        return (a>b) * 1.0
    else:
        return -1

#比较两个矩阵大小，返回01矩阵小
def mat_small(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        return (a<b) * 1.0
    else:
        return -1

#比较两个矩阵大小，返回较大值
def mat_keep(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and b_type: #两个df
        con_a = np.any(a<0) | np.any(a>1) | np.any((a>0) & (a<1)) 
        con_b = np.any(b<0) | np.any(b>1) | np.any((b>0) & (b<1)) 
        if ~con_a or ~con_b:
            return a * b
        else:
            return -1
    else:
        return -1

#返回一组数与[0,1,2,3,4,5,6,...]的回归beta       
def regbeta(a, b):
    a_type = isinstance(a, np.ndarray)
    b_type = isinstance(b, np.ndarray)
    if a_type and ~b_type:
        t = np.array([np.arange(a.shape[0])])
        t = np.repeat(t,a.shape[1],axis = 0).T
        return rw_beta(a, t, b)
    elif ~a_type and b_type:   
        t = np.array([np.arange(b.shape[0])])
        t = np.repeat(t,b.shape[1],axis = 0).T
        return rw_beta(b, t, a)  
    elif a_type and b_type:  
        t = np.array([np.arange(a.shape[0])])
        t = np.repeat(t,a.shape[1],axis = 0).T
    else:
        return -1             
    
    
#==========================================因子============================================================
def factor1(A,B,b,c):
    tmp1 = sign(delta(A,1))
    return add(add(tmp1,delay(tmp1,1)),delay(tmp1,2))*rw_ma(B,b)/rw_ma(B,b*c)

def factor2(A, B, C, a, b, c, d, e):
    tmp1 = rw_ma(B,a)
    tmp1 = rw_corr(A, tmp1, b)
    tmp1 = rw_rank(tmp1, c)   
    tmp2 = delta(C,d)
    tmp2 = rw_rank(tmp2,e)
    return add(tmp1,tmp2)

def factor3(close, a):
    return delta(close,a)

def factor4(A,B,C,a):
    tmp = sub(A,B)
    tmp = tmp-sub(B,C)  
    tmp = div(tmp,sub(A,C))
    return delta(tmp,a)

def factor5(A,a):
    return delta_perc(A,a)

def factor6(A,B,a,b):
    tmp = rw_ma(A,a)
    tmp = mul(tmp,rw_ma(B,a))
    return delta_perc(tmp,b)

def factor7(A,a,b):
    return div(A, add(rw_ma(A,a), rw_ma(A,b)))

def factor8(A,a):
    return div(A, rw_ma(A, a))

def factor9(A, B, a, b):
    return div(div(A, rw_ma(A, a)),delta_perc(B,b))

def factor10(A,B,C,a):
    tmp = rw_max(A,a)
    tmp = rank(A-tmp)
    return div(tmp,rank(B+C))
    #return div(rank(A-rw_max(A,a),b),rank(B+C,c))

def factor11(A,B,C,a):
    tmp = rw_min(A,a)
    tmp = rank(A-tmp)
    return div(tmp,rank(B+C))
    #return div(rank(A-rw_min(A,a),b),rank(B+C,c))

def factor12(A, B, a, b):
    tmp1 = rw_ma(A, a)
    tmp1 = div(A, tmp1)
    tmp1 = rank(tmp1)
    tmp2 = delta_perc(B,b)
    tmp2 = rank(tmp2)
    return div(tmp1,tmp2)
    #return div(rank(div(A, rw_ma(A, a)),b), rank(delta_perc(B,c),d))

def factor13(A, B, a, b):
    tmp = rw_ma(A,a)
    tmp1 = rw_ma(B,a*b)
    tmp = div(tmp,tmp1)
    return tmp
    #return div(rw_ma(A,a),rw_ma(B,a*b))

def factor14(A,B,C,a,b):
    tmp = delay(A,a)
    tmp1 = tmp - B
    tmp1 = maxm(tmp1,0)
    tmp1 = rw_ma(tmp1,b)
    tmp2 = C-tmp
    tmp2 = maxm(tmp2,0)
    tmp2 = rw_ma(tmp2,b)
    return div(tmp1, tmp2)
    #div(rw_ma(maxm(delay(A,a)-B,0),b),rw_ma(maxm(C-delay(A,a),0),b))

def factor15(A,a,b):
    tmp = delta(A,a)
    tmp1 = maxm(tmp, 0)
    tmp1 = rw_ma(tmp1,b)
    tmp2 = abso(tmp)
    tmp2 = rw_ma(tmp2,b)
    return div(tmp1, tmp2)
    #div(rw_ma(maxm(delta(A,a),0),b), rw_ma(abso(delta(A,a)),b))

def factor16(A,B,C,a):
    tmp1 = sub(A,B)
    tmp1 = rw_ma(tmp1,a)
    tmp2 = sub(B,C)
    tmp2 = rw_ma(tmp2,a)
    return div(tmp1, tmp2)
    #div(rw_ma(sub(A,B),a),rw_ma(sub(B,C),a))


def factor17(A,B,C):
    return div(sub(A,B),C)

def factor18(A,B,a,b,c):
    tmp1 = rw_ma(A,a)
    tmp1 = sub(A,tmp1)
    tmp2 = B-rw_ma(A, b)
    tmp2 = abso(tmp2)
    tmp2 = rw_ma(tmp2, b*c)
    return div(tmp1, tmp2)
    #div(sub(A,rw_ma(A,a)),rw_ma(abso(B-rw_ma(A, b)),b*c))

def factor19(A,B,C,a,b):
    tmp1 = delta(A,a)
    tmp1 = rank(tmp1)
    tmp2 = delta(B,b)
    tmp2 = rank(tmp2) 
    tmp2 = sub(tmp1,tmp2)
    return div(tmp2,rank(C))
    #div(sub(rank(delta(A,a),b),rank(delta(B,c),d)),rank(C,e))

def factor20(A,B,a,b):
    tmp1 = rw_ma(B,b)
    tmp1 = mat_big(B, tmp1)
    return mat_keep(delta_perc(A,a),tmp1)
    #mat_keep(delta_perc(A,a),mat_big(B, rw_ma(B,b)))

def factor21(A,B,a,b):
    tmp1 = rw_ma(B,b)
    tmp1 = mat_small(B, tmp1)
    return mat_keep(delta_perc(A,a),tmp1)
    #mat_keep(delta_perc(A,a),mat_small(B, rw_ma(B,b)))

def factor22(A,B,C,a,b):
    tmp1 = delay(B,b)
    tmp1 = mat_big(A, tmp1)
    return mat_keep(delta_perc(C,a),tmp1)
    #mat_keep(delta_perc(C,a),mat_big(A, delay(B,b)))

def factor23(A,B,C,a,b):
    tmp1 = delay(B,b)
    tmp1 = mat_small(A, tmp1)
    return mat_keep(delta_perc(C,a),tmp1)
    #mat_keep(delta_perc(C,a),mat_small(A, delay(B,b)))
  
def factor24(A,volume_adj,a):
    return mul(delta_perc(A,a),volume_adj)

def factor25(A,B,a,b):
    tmp = rw_ma(A, a)
    tmp = div(A, rw_ma(A, a))
    return mul(tmp,delta_perc(B,b))
    #mul(div(A, rw_ma(A, a)),delta_perc(B,b))

def factor26(A,B,C,D):
    tmp1 = rank(A-delay(B,1))
    tmp2 = rank(A)
    tmp2 = tmp2 * tmp2
    return mul(tmp1,tmp2)
    #mul(mul(rank(A-delay(B,1)),rank(A)),rank(A))

def factor27(A,B,a,b):
    tmp1 = rw_ma(A, a)
    tmp1 = div(A, tmp1)
    tmp1 = rank(tmp1)
    tmp2 = delta_perc(B,b)
    tmp2 = rank(delta_perc(B,b))
    tmp1 = mul(tmp1,tmp2)
    return tmp1
    #return mul(rank(div(A, rw_ma(A, a))), rank(delta_perc(B,b)))


def factor28(A,B,C,a,b,c):
    tmp1 = rw_std(A,a)
    tmp1 = rank(tmp1)
    tmp2 = rw_corr(B,C,b)
    tmp2 = delta(tmp2,c)
    return mul(tmp1,tmp2)
    #mul(rank(rw_std(A,a)),delta(rw_corr(B,C,b),c))

def factor29(A,B,C,a,b):
    return mul(rw_std(A,a),rw_corr(B,C,b))

def factor30(A,B,C,D,a,b):
    tmp1 = abso(A-B)
    tmp1 = rw_std(tmp1,a)
    tmp2 = rw_corr(C,D,b)
    return mul(tmp1,tmp2)
    #mul(rw_std(abso(A-B),a),rw_corr(C,D,b))

def factor31(volume_adj,B,a):
    return mul(rw_std(volume_adj, a), B)

def factor32(A,B,a,b):
    tmp = delta_perc(A,a)
    tmp = rank(tmp)
    return tmp * rw_rank(B,b)
    #return rank(delta_perc(A,a)) * rw_rank(B,b)

def factor33(A,B,C,a):
    tmp1 = delay(B-C,1)
    tmp1 = rw_corr(A, tmp1, a)
    tmp1 = rank(tmp1)
    tmp1 = tmp1 + rank(B-C)
    return tmp1
    #rank(rw_corr(A,  delay(B-C,1),a)) + rank(B-C)

def factor34(A,B,a,b,c):
    tmp1 = delta_perc(A,a)
    tmp1 = rw_ma(tmp1,b)
    tmp1 = rank(tmp1)
    tmp2 = rw_std(B,c)
    tmp2 = rank(tmp2)
    return tmp1-tmp2
    #rank(rw_ma(delta_perc(A,a),b))-rank(rw_std(B,c))

def factor35(A,B):
    tmp1 = sub(A,B)
    tmp1 = rank(tmp1)
    tmp2 = add(A,B)
    tmp2 = rank(tmp2)
    return tmp1/tmp2
    #rank(sub(A,B))/rank(add(A,B))

def factor36(A,a):
    return regbeta(A,a)

def factor37(A,a):
    return regbeta(rank(A),a)

def factor38(A,a,b):
    return regbeta(rw_ma(A,a),b)

def factor39(A,a):
    return rw_argmaxmin(A,a)

def factor41(A,B,a):
    return rw_beta(A,B,a)

def factor42(A,B,a,b,c):
    return rw_beta(delta(A,a),delta(B,b),c)

def factor43(A,B,a,b,c):
    return rw_beta(delta(A,a),rw_ma(B,b),c)

def factor44(A,B,a,b,c):
    return rw_beta(rw_ma(A,a),rw_ma(B,b),c)

def factor45(A,B,a,b):
    return rw_corr(A, delay(B,b),a)

def factor46(A,B,C,a,b):
    tmp = B-C
    tmp = delay(tmp,b)
    return rw_corr(A, tmp ,a)
    #return rw_corr(A, delay(B-C,b),a)

def factor47(A,B,a,b):
    return rw_corr(A, rw_ma(B,a),b)

def factor48(A,B,a):
    return rw_corr(A,B,a)

def factor49(A,B,a):
    return rw_corr(rank(A),rank(B),a)

def factor50(A,B,a,b):
    tmp = rank(B)
    tmp = rw_ma(tmp,b)
    return rw_corr(rank(A), tmp, a)
    #return rw_corr(rank(A),rw_ma(rank(B),b),a)

def factor51(A,B,C,a,b):
    tmp1 = delta(C,a)
    tmp1 = rank(tmp1)
    tmp2 = sub(A,B)
    tmp2 = div(tmp2,B)
    return rw_corr(tmp1,tmp2,b)
    #rw_corr(rank(delta(C,a)),rank(div(sub(A,B),B)),b)

def factor52(A,B,C,a,b):
    tmp1 = delta_perc(C,a)
    tmp1 = rank(tmp1)
    tmp2 = sub(A,B)
    tmp2 = div(tmp2,B)
    return rw_corr(tmp1,tmp2,b)
    #rw_corr(rank(delta_perc(C,a)),rank(div(sub(A,B),B)), b)

def factor53(A,B,a,b,c):
    return rw_corr(rw_rank(A,a),rw_rank(B,b), c)

def factor54(volume_adj, a, b, c): 
    tmp = rw_ma(volume_adj, c)
    tmp = rw_argmaxmin(tmp, b)
    return rw_ma(tmp, a)
    #rw_ma( rw_argmaxmin(rw_ma(volume_adj, c),b), a) 

def factor55(close, a):
    return rw_ma(close,a)

def factor56(A,a):
    tmp = delta(A,1)
    return rw_ma(tmp,a)/rw_std(tmp,a)
    #rw_ma(delta(A,1),a)/rw_std(delta(A,1),a)

def factor57(close,a,b):
    return rw_ma(delta(close,a),b)

def factor58(A,a,b):
    return rw_ma(delta_perc(A,a),b)

def factor59(A,B,a):
    tmp = delta(A,1)
    tmp = abso(tmp)
    tmp = div(tmp,B)
    return rw_ma(tmp,a)/rw_std(tmp,a)
    #rw_ma(div(abso(delta(A,1)),B),a)/rw_std(div(abso(delta(A,1)),B),a)

def factor60(A,B,a):
    tmp = delta(A,1)
    tmp = div(tmp,B)
    return rw_ma(tmp,a)/rw_std(tmp,a)
    #rw_ma(div(delta(A,1),B),a)/rw_std(div(delta(A,1),B),a)

def factor61(A,B,a,b,c):
    tmp0 = delta_perc(A,a)
    tmp1 = mat_big(B, rw_ma(B,b))
    tmp1 = mat_keep(tmp0,tmp1)
    tmp2 = mat_small(B, rw_ma(B,b))
    tmp2 = mat_keep(tmp0,tmp2) 
    tmp0 = div(tmp1,tmp2)
    return rw_ma(tmp0,c)
    #rw_ma(div(mat_keep(delta_perc(A,a),mat_big(B, rw_ma(B,b))),mat_keep(delta_perc(A,a),mat_small(B, rw_ma(B,b)))),c)

def factor62(A,B,C,a):
    tmp = sub(A,B)
    tmp = div(tmp,C)
    return rw_ma(tmp, a)
    #return rw_ma(div(sub(A,B),C), a)

def factor64(A,B,C,a):
    tmp = sub(A,B)
    tmp = tmp - sub(B,C)
    tmp = div(tmp,sub(A,C))
    return rw_ma(tmp,a)
    #rw_ma(div(sub(A,B)-sub(B,C),sub(A,C)),a)

def factor65(A,a,b):
    tmp = delta(A,a)
    tmp = maxm(tmp,0)
    return rw_ma(tmp,b)
    #rw_ma(maxm(delta(A,a),0),b)

def factor66(A,B,C,D,a):
    tmp = mul(2,A)
    tmp = sub(tmp,add(B,C))
    tmp = div(tmp,sub(B,C))
    tmp = mul(tmp,D)
    return rw_ma(tmp,a)
    #rw_ma(mul(div(sub(mul(2,A),add(B,C)),sub(B,C)),D),b)

def factor67(A,a,b):
    tmp = delta_perc(A,1)
    tmp1 = rw_ma(tmp,a)
    tmp2 = rw_std(tmp,a)
    tmp = mul(tmp1,tmp2)
    tmp = rw_ma(tmp,b)
    return tmp
    #rw_ma(mul(rw_ma(delta_perc(A,1),a),rw_std(delta_perc(A,1),a)),b)

def factor68(A,B,a,b):
    tmp = rw_corr(rank(A),rank(B),a)
    tmp = rank(tmp)
    tmp = rw_ma(tmp,b)
    return tmp
    #rw_ma(rank(rw_corr(rank(A),rank(B),a)),b)

def factor69(A,B,a,b):
    return rw_ma(rw_beta(A,B,a),b)

def factor70(A,B,a,b):
    return rw_ma(rw_corr(rank(A),rank(B),a),b)

def factor71(A,a,b):
    return rw_ma(rw_maxmin_norm(A,a), b)

def factor72(A,a,b):
    tmp = delta(A,a)
    tmp1 = maxm(tmp,0)
    tmp2 = minm(tmp,0)
    tmp = sub(tmp1,tmp2)
    tmp = rw_ma(tmp,b)
    return tmp
    #rw_ma(sub(maxm(delta(A,a),0), minm(delta(A,a),0)),b)

def factor73(A, volume_adj, a):
    return rw_max(mul(volume_adj, A),a)

def factor74(A,B,a,b,c,d):
    tmp = rw_rank(A,a)
    tmp = rw_corr(tmp,rw_rank(B,b),c)
    tmp = rw_max(tmp,d)
    return tmp
    #rw_max(rw_corr(rw_rank(A,a),rw_rank(B,b), c),d)

def factor75(A,a,b):
    return rw_maxmin_norm(delta_perc(A,a),b)

def factor76(A,B,a,b,c,d):
    tmp = rw_rank(A,a)
    tmp = rw_corr(tmp,rw_rank(B,b),c)
    tmp = rw_min(tmp,d)
    return tmp
    #rw_min(rw_corr(rw_rank(A,a),rw_rank(B,b), c),d)

def factor77(A,a,b):
    return rw_rank(delta(A,b),a)

def factor78(A,a,b):
    tmp = rw_ma(A,b)
    tmp = div(A, tmp)
    return rw_rank(tmp,a)
    #rw_rank(div(A, rw_ma(A,b)),a)

def factor79(A,B,a,b,c,d):
    tmp = rw_ma(A,b)
    tmp = div(A, tmp)
    tmp = rw_rank(tmp,a)
    tmp1 = delta(B,c)
    tmp1 = rw_rank(delta(B,c),d)
    tmp = tmp * tmp1
    return tmp
    #rw_rank(div(A, rw_ma(A,b)),a) * rw_rank(delta(B,c))

def factor80(A,B,a):
    return rw_std(abso(A-B),a)

def factor81(A,B,a,b):
    return rw_std(rw_beta(A,B,a),b)

def factor82(volume_adj,a):
    return rw_std(volume_adj,a)

def factor83(A,a,b):
    return sub(rw_ma(A,a),rw_ma(A,a*b))

def factor84(A,B,C,a,b):
    tmp = mul(A,B)
    tmp = div(mul(A,B),sqr(C))
    return sub(rw_ma(tmp,a),rw_ma(tmp,a*b))
    #sub(rw_ma(div(mul(A,B),sqr(C)),a), rw_ma(div(mul(A,B),sqr(C)),a*b))

def factor85(A,B,a,b):
    return sub(rw_maxmin_norm(A,a), rw_maxmin_norm(B,a*b))

def factor86(A,B,C,D):
    tmp1 = rank(A-delay(B,1))
    tmp2 = rank(A-delay(C,1))
    tmp1 = mul(tmp1,tmp2)
    tmp2 = rank(A-delay(D,1))
    return mul(tmp1,tmp2)
    #mul(mul(rank(A-delay(B,1)),rank(A-delay(C,1))),rank(A-delay(D,1)))