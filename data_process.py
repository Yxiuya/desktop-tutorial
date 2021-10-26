# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 08:41:51 2021

@author: liuzhenguo
"""

import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import warnings
from datetime import timedelta,date
from sklearn.linear_model import Ridge

path='/lzg/competition/train/风场2/x25/'
path_wea='/lzg/competition/train/风场2/weather.csv'
file_name=os.listdir(path)
file_name.remove('data.csv')
# data=pd.read_csv(path+file_name[0])
# data_wea=pd.read_csv(path_wea)

def date_process(data):
    data['date']=pd.to_datetime(data['time'])
    
    data['year']=pd.to_numeric(data['date'].dt.year.astype('float'),downcast='float')
    data['month']=pd.to_numeric(data['date'].dt.month.astype('float'),downcast='float')
    data['day']=pd.to_numeric(data['date'].dt.day.astype('float'),downcast='float')
    data['hour']=pd.to_numeric(data['date'].dt.hour.astype('float'),downcast='float')
    # data['minute']=pd.to_numeric(data['date'].dt.minute.astype('float'),downcast='float')
    # data['second']=pd.to_numeric(data['date'].dt.second.astype('float'),downcast='float')
    return data

data=pd.DataFrame()
# emp_df=list(pd.read_csv('emp_df.csv')['0'])
start_time=date(2017,1,1)
end_time=date(2018,12,31)
date_time=pd.DataFrame(pd.date_range(start=start_time,end=end_time,freq='30S'),columns=['time'])
date_time['time']=pd.to_datetime(date_time['time'])


for i in tqdm(range(len(file_name))):
    data_tem=pd.read_csv(path+file_name[i])
    data=pd.concat([data,data_tem],axis=0,ignore_index=True)
data['time']=pd.to_datetime(data['time'])
data_tem=pd.merge(date_time,data,on='time',how='outer')
type(data.loc[0,'time'])
data_tem=data_tem.sort_values('time')
data_tem.to_csv(path+'data.csv',index=False)
# emp_df=pd.DataFrame(emp_df)
# emp_df.to_csv('emp_df.csv',index=False)   
# data.to_csv(path+'data.csv',index=False)

# data=pd.read_csv(path+'data.csv')
# data=date_process(data)
# data_wea=pd.read_csv(path_wea)


# data_tem=pd.read_csv(path+'2018-08-15.csv')
# data_tem['time']=pd.to_datetime(data_tem['time'])
# date_time=pd.date_range(start=date(2018,8,15),periods=3600*24/30,freq='30S')
# date_time=pd.DataFrame(date_time,columns=['time'])
# date_time['time']=pd.to_datetime(date_time['time'])
# data_tem1=pd.merge(date_time,data_tem,on='time',how='outer')

