#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:33:39 2021

@author: root
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
from sklearn.model_selection import train_test_split
import joblib
from sklearn.utils import shuffle
from PyEMD import EEMD, Visualisation
import warnings
warnings.filterwarnings('ignore')

def season_lgb(x):
    if x=='春':
        return 0
    if x=='夏':
        return 1
    if x=='秋':
        return 3
    if x=='冬':
        return 4
def date_process(data):
    data['time']=pd.to_datetime(data['time'])
    data['year']=pd.to_numeric(data['time'].dt.year.astype('float'),downcast='float')
    data['month']=pd.to_numeric(data['time'].dt.month.astype('float'),downcast='float')
    data['day']=pd.to_numeric(data['time'].dt.day.astype('float'),downcast='float')
    data['hour']=pd.to_numeric(data['time'].dt.hour.astype('float'),downcast='float')
    # data['minute']=pd.to_numeric(data['date'].dt.minute.astype('float'),downcast='float')
    # data['second']=pd.to_numeric(data['date'].dt.second.astype('float'),downcast='float')
    return data
def eemd_fs(x):
    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(12345)
    E_IMFs = eemd.eemd(x,max_imf=4)
    # print(E_IMFs.size)
    # print(imfs)
    return E_IMFs.T
#读取数据与处理标签、日期
fc=1
result=pd.DataFrame()
if fc ==1:
    x_num=[26,51]
else:
    x_num=[25,50]
for j in tqdm(range(x_num[0],x_num[1])):
    # j=26
    path='/lzg/competition/test/风场'+str(fc)+'/x'+str(j)+'/'
    path_wea='/lzg/competition/test/风场'+str(fc)+'/weather.csv'
    data_wea=pd.read_csv(path_wea)
    data_wea=data_wea.drop('风速',axis=1)
    if fc==1:
        name_mmx='mmx_wind_dir'
    else:
        name_mmx='mmx_wind_dir2'
    mmx=joblib.load(name_mmx)
    data_wea['风向']=mmx.fit_transform(data_wea['风向'].values.reshape(-1,1))
    file_name=os.listdir(path)
    data=pd.DataFrame()
    data_train=pd.DataFrame()
    for i in range(len(file_name)):
        data_tem=pd.read_csv(path+file_name[i])
        data_tem['文件名']=file_name[i]
        ee=eemd_fs(data_tem['风向'].values)
        eee=pd.DataFrame(ee)
        if len(list(eee.columns))<5:
            for l in range(5-len(list(eee.columns))):
                e_tem=pd.DataFrame([np.nan for m in range(120)])
                eee=pd.concat([eee,e_tem],axis=1)
        eee.columns=['emd_1','emd_2','emd_3','emd_4','emd_5']
        data_tem=pd.concat([data_tem,eee],axis=1)
        data_tem['dir_diff']=data_tem['风向'].diff()
        data_tem['dir_diff2']=data_tem['dir_diff'].diff()
        for k in range(1,11):
            data_tem['fx_qian_'+str(k+20)]=data_tem['风向'].shift(k)
        remove_col=['风速','文件名','time']
        fea=list(data_tem.columns)
        for k in remove_col:
            fea.remove(k)
        for k in range(10):
            sd=file_name[i].split('.')[0]
            sk=-1*k
            if k==0 :
                name='wind_dir'
            else:
                name='wind_dir_qian'+str(k)
            wed_t=float(data_wea.query('时段 == @sd and 时刻 == @sk')['风向'].values)
            wed=[wed_t for l in range(120)]
            data_tem[name]=wed
        # data_tem['dir_qian30']=data_tem['风向'].shift(-30)
        data_train_tem=data_tem.copy()
        data_train_tem['y_test']=data_tem['风向'].shift(-20)
        
        data_train_tem=data_train_tem.dropna()
        # data_tem=data_tem.dropna()
        if len(data_tem) < 118 :
            print(file_name[i])
        data=pd.concat([data,data_tem],axis=0,ignore_index=True)
        data_train=pd.concat([data_train,data_train_tem],axis=0,ignore_index=True)
    data['season']=data['文件名'].apply(lambda x:x.split('_')[0]).astype('category')
    data_train['season']=data_train['文件名'].apply(lambda x:x.split('_')[0]).astype('category')
    # data['y_test']=data['风速'].shift(-20)
    # data=data.dropna()
    # data=date_process(data)
    
    #聚合特征
    def ex_fea(data,tr='no_train'):
        group_feats = []
        for f in tqdm(fea):
            data['MDH_{}_medi'.format(f)] = data.groupby('文件名')[f].transform('median')
            data['MDH_{}_mean'.format(f)] = data.groupby('文件名')[f].transform('mean')
            data['MDH_{}_max'.format(f)] = data.groupby('文件名')[f].transform('max')
            data['MDH_{}_min'.format(f)] = data.groupby('文件名')[f].transform('min')
            data['MDH_{}_std'.format(f)] = data.groupby('文件名')[f].transform('std')
            data['MDH_{}_minmax'.format(f)]=data['MDH_{}_max'.format(f)]-data['MDH_{}_min'.format(f)]
            
            group_feats.append('MDH_{}_medi'.format(f))
            group_feats.append('MDH_{}_mean'.format(f))
            group_feats.append('MDH_{}_max'.format(f))
            group_feats.append('MDH_{}_min'.format(f))
            group_feats.append('MDH_{}_std'.format(f))
        #交叉特征
        for f1 in tqdm(fea + group_feats):
        
            for f2 in fea + group_feats:
                if f1 != f2:
                    colname = '{}_{}_ratio'.format(f1, f2)
                    data[colname] = data[f1].values / data[f2].values
        
        data = data.fillna(method='bfill')
        if tr=='train':
            if fc==1:
                col_name='col_dir_train.csv'
            else:
                col_name='col_dir2_train.csv'
            c=pd.read_csv(col_name,encoding='gbk')
        else:
            if fc==1:
                col_name='col_dir.csv'
            else:
                col_name='col_dir2.csv'
            c=pd.read_csv(col_name,encoding='gbk')
        # c.to_csv('col.csv',index=False,encoding='gbk')
        data=data[c['0']]
        #离散化
        '''
        for f in fea:
            data[f + '_20_bin'] = pd.cut(data[f], 20, duplicates='drop').apply(lambda x: x.left).astype(float)
            data[f + '_50_bin'] = pd.cut(data[f], 50, duplicates='drop').apply(lambda x: x.left).astype(float)
            data[f + '_100_bin'] = pd.cut(data[f], 100, duplicates='drop').apply(lambda x: x.left).astype(float)
            data[f + '_200_bin'] = pd.cut(data[f], 200, duplicates='drop').apply(lambda x: x.left).astype(float)
        
        for f1 in tqdm(
                [i+'_20_bin' for i in fea]):
            for f2 in fea :
                data['{}_{}_medi'.format(f1, f2)] = data.groupby([f1])[f2].transform('median')
                data['{}_{}_mean'.format(f1, f2)] = data.groupby([f1])[f2].transform('mean')
                data['{}_{}_max'.format(f1, f2)] = data.groupby([f1])[f2].transform('max')
                data['{}_{}_min'.format(f1, f2)] = data.groupby([f1])[f2].transform('min')
        for f1 in tqdm(
                 [i+'_50_bin' for i in fea]):
            for f2 in fea:
                data['{}_{}_medi'.format(f1, f2)] = data.groupby([f1])[f2].transform('median')
                data['{}_{}_mean'.format(f1, f2)] = data.groupby([f1])[f2].transform('mean')
                data['{}_{}_max'.format(f1, f2)] = data.groupby([f1])[f2].transform('max')
                data['{}_{}_min'.format(f1, f2)] = data.groupby([f1])[f2].transform('min')
        for f1 in tqdm(
                 [i+'_100_bin' for i in fea]):
            for f2 in fea:
                data['{}_{}_medi'.format(f1, f2)] = data.groupby([f1])[f2].transform('median')
                data['{}_{}_mean'.format(f1, f2)] = data.groupby([f1])[f2].transform('mean')
                data['{}_{}_max'.format(f1, f2)] = data.groupby([f1])[f2].transform('max')
                data['{}_{}_min'.format(f1, f2)] = data.groupby([f1])[f2].transform('min')
        for f1 in tqdm(
                 [i+'_200_bin' for i in fea]):
            for f2 in fea:
                data['{}_{}_medi'.format(f1, f2)] = data.groupby([f1])[f2].transform('median')
                data['{}_{}_mean'.format(f1, f2)] = data.groupby([f1])[f2].transform('mean')
                data['{}_{}_max'.format(f1, f2)] = data.groupby([f1])[f2].transform('max')
                data['{}_{}_min'.format(f1, f2)] = data.groupby([f1])[f2].transform('min')
        '''
        return data
    data=ex_fea(data)
    data_train=ex_fea(data_train,'train')
    if fc==1:
        fs_base=0.15
    else:
        fs_base=0.086
    data_train=data_train.query('风速 > @fs_base')
    drop_columns = ["time",'文件名','风速']
    # data_test=data.drop(drop_columns,axis=1)
    # data_test.to_csv('test.csv',index=False)
    # data_y_test=data['y_test']
    # data_y_test.to_csv('test_y.csv',index=False)
    # te=pd.read_csv('test.csv')
    # te_y=pd.read_csv('test_y.csv')
    # te=te.drop('year',axis=1)
    data_pre=data.drop(drop_columns,axis=1)
    data_pre['season']=data_pre['season'].apply(lambda x:season_lgb(x)).astype('category')
    if fc==1:
        fs_base=0.15
    else:
        fs_base=0.086
    data_train=data_train.query('风速 > @fs_base')
    data_train=shuffle(data_train).reset_index(drop=True)
    data_tr=data_train.drop(drop_columns,axis=1)
    x_train=data_tr.drop('y_test',axis=1)
    x_train['season']=x_train['season'].apply(lambda x:season_lgb(x)).astype('category')
    y_train=data_tr['y_test']
    # model_lgb = lgb.Booster(model_file='lgbm.txt')
    train_matrix = lgb.Dataset(x_train, label=y_train)
    params = {
                'boosting_type': 'gbdt',
                'objective': 'mse',
                'min_child_weight': 5,
                'num_leaves': 2 ** 8,
                'feature_fraction': 0.3,
                'bagging_fraction': 0.3,
                'bagging_freq': 1,
                'learning_rate': 0.01,
                'seed': 2020,
                'device':'gpu',
                'gpu_platform_id':0,
                'gpu_device_id':0,
                
            }
    if fc==1:
        name_lgb='lgbm_dir.txt'
        name_cat='cat_dir'
    else:
        name_lgb='lgbm_dir2.txt'
        name_cat='cat_dir2'
    # gbm = lgb.train(params,train_matrix,
    #                 num_boost_round=10,
    #                 init_model=name_lgb,categorical_feature=['season'])
    # model_lgb_pre=gbm.predict(data_pre)
    
    model_cat = CatBoostRegressor()
    model_cat.load_model(name_cat)
    params_cat = {'learning_rate': 0.0001, 'depth': 10, 'l2_leaf_reg': 1, 'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False,
                  'use_best_model': True,'task_type' : 'GPU'}
    
    nums = int(x_train.shape[0] * 0.80)
    trn_x, trn_y, val_x, val_y = x_train[:nums], y_train[:nums], x_train[nums:], y_train[nums:]
    model_cat=model_cat.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=['season'], use_best_model=True, verbose=500)
    model_cat_pre=model_cat.predict(data_pre)
    # model_sgd = joblib.load('sgb.pkl')
    # model_ridge = joblib.load('ridge.pkl')
    # mmx=MinMaxScaler((0,1))
    # data_pre_mmx=mmx.fit_transform(data_pre)
    # model_sgd_pre=model_sgd.predict(data_pre_mmx)
    # model_lgb_pre=model_lgb.predict(data_pre)
    tem={'风向':model_cat_pre,'time':data['time'],'时段':data['文件名'].apply(lambda x: x.split('.')[0])}
    model_cat_pre=pd.DataFrame(tem)
    # plt.plot(model_lgb_pre)
    ans_tem=pd.DataFrame()
    if fc==2:
        if j==40:
            ans_tem['风机']=['x'+str(j) for k in range(1460)]
        elif j==44:
            ans_tem['风机']=['x'+str(j) for k in range(960)]
        else:
            ans_tem['风机']=['x'+str(j) for k in range(1600)]
    else:      
        ans_tem['风机']=['x'+str(j) for k in range(1600)]
    # ans_tem['风机']=['x'+str(j) for k in range(len(model_cat_pre.query('time >= 3030 and time <= 3600')['时段']))]
    ans_tem['时段']=list(model_cat_pre.query('time >= 3030 and time <= 3600')['时段'])
    ans_tem['时刻']=list(model_cat_pre.query('time >= 3030 and time <= 3600')['time']-3000)
    ans_tem['风向']=list(model_cat_pre.query('time >= 3030 and time <= 3600')['风向'])
    result=pd.concat([result,ans_tem],axis=0,ignore_index=True)
result.to_csv('result_风向.csv',index=False)
