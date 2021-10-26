#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:21:20 2021

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
from PyEMD import EEMD, Visualisation
import datetime
import warnings
warnings.filterwarnings('ignore')


def date_process(data):
    data['time']=pd.to_datetime(data['time'])
    data['year']=pd.to_numeric(data['time'].dt.year.astype('float'),downcast='float')
    data['month']=pd.to_numeric(data['time'].dt.month.astype('float'),downcast='float')
    data['day']=pd.to_numeric(data['time'].dt.day.astype('float'),downcast='float')
    data['hour']=pd.to_numeric(data['time'].dt.hour.astype('float'),downcast='float')
    # data['minute']=pd.to_numeric(data['date'].dt.minute.astype('float'),downcast='float')
    # data['second']=pd.to_numeric(data['date'].dt.second.astype('float'),downcast='float')
    return data
def season(x):
    if x==1 or x==2 or x==3:
        return '春'
    if x==4 or x==5 or x==6:
        return '夏'
    if x==7 or x==8 or x==9:
        return '秋'
    if x==10 or x==11 or x==12:
        return '冬'
def season_lgb(x):
    if x=='春':
        return 0
    if x=='夏':
        return 1
    if x=='秋':
        return 3
    if x=='冬':
        return 4
#读取数据与处理标签、日期
fc=1#风场号

path_wea='/lzg/competition/train/风场'+str(fc)+'/weather.csv'
data_wea=pd.read_csv(path_wea)
data_wea=data_wea.drop('wind_spd',axis=1)

mmx=MinMaxScaler((0,1))
data_wea['wind_dir']=mmx.fit_transform(data_wea['wind_dir'].values.reshape(-1,1))
if fc==1:
    name_mmx='mmx_wind_dir'
else:
    name_mmx='mmx_wind_dir2'
joblib.dump(mmx,name_mmx)

if fc==1:
    fs_base=0.15
else:
    fs_base=0.086
def read_data(fj):
    path='/lzg/competition/train/风场'+str(fc)+'/x'+str(fj)+'/'
    data=pd.read_csv(path+'data.csv')
    data['y_test']=data['风向'].shift(-20)
    for i in range(1,11):
        data['fx_qian_'+str(i+20)]=data['风向'].shift(i)
    # data['dir_qian30']=data['风向'].shift(-30)
    #差分特征
    data['dir_diff']=data['风向'].diff()
    data['dir_diff2']=data['dir_diff'].diff()
    data=data.dropna()
    data=data.query('风速 > @fs_base')
    return data
if fc==1:
    data1=read_data(26)
    data2=read_data(27)
    data3=read_data(28)
    data4=read_data(29)
    data=pd.concat([data1,data2,data3,data4],axis=0,ignore_index=True)
else:
    data1=read_data(25)
    data2=read_data(26)
    data=pd.concat([data1,data2],axis=0,ignore_index=True)
data=date_process(data)
data_wea=date_process(data_wea)
data_wea_tem=data_wea.drop('time',axis=1)
data=pd.merge(data,data_wea_tem,left_on=['year','month','day','hour'],right_on=['year','month','day','hour'],how='outer')
remove_col=['year','month','day','hour','time','y_test','wind_dir','风速']
fea=list(data.columns)
for i in remove_col:
    fea.remove(i)
for i in range(1,10):
    data_wea['time']=data_wea['time']+datetime.timedelta(hours=1)
    col=['time','wind_dir_qian'+str(i),'year','month','day','hour']
    data_wea=date_process(data_wea)
    data_wea.columns=col
    data_wea_tem=data_wea.drop('time',axis=1)
    data=pd.merge(data,data_wea_tem,left_on=['year','month','day','hour'],right_on=['year','month','day','hour'],how='outer')
data=data.dropna()
data_1=data.query('year == 2018 and month == 1 and day == 3 and hour == 19')
data=data.drop(data_1.index)
data['season']=data['month'].apply(lambda x:season(x)).astype('category')
def eemd_fs(x):
    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(12345)
    E_IMFs = eemd.eemd(x,max_imf=4)
    # print(E_IMFs.size)
    # print(imfs)
    return E_IMFs.T
'''
ee=[]
ind=[]
for i in tqdm(data.groupby(['year','month', 'day', 'hour'])['风向']):
    if len(i[1].values)<10:
        ind.extend(i[1].index)
data=data.drop(ind)
for i in tqdm(data.groupby(['year','month', 'day', 'hour'])['风向']):
    ee_tem=eemd_fs(i[1].values)
    ee.extend(ee_tem)        
eee=pd.DataFrame(ee)
eee.columns=['emd_1','emd_2','emd_3','emd_4','emd_5']
eee.to_csv('eemd_dir2.csv',index=False)
'''
if fc==1:
    data_emd=pd.read_csv('/lzg/competition/eemd_data/eemd_dir.csv')
else:
    data_emd=pd.read_csv('/lzg/competition/eemd_data/eemd_dir2.csv')
data=data.reset_index(drop=True)
data=pd.concat([data,data_emd],axis=1)
fea=fea+['emd_1','emd_2','emd_3','emd_4','emd_5']
data=data.dropna()
#聚合特征
group_feats = []
for f in tqdm(['变频器电网侧有功功率', '外界温度','风向','dir_diff','dir_diff2']):
    data['MDH_{}_medi'.format(f)] = data.groupby(['year','month', 'day', 'hour'])[f].transform('median')
    data['MDH_{}_mean'.format(f)] = data.groupby(['year','month', 'day', 'hour'])[f].transform('mean')
    data['MDH_{}_max'.format(f)] = data.groupby(['year','month', 'day', 'hour'])[f].transform('max')
    data['MDH_{}_min'.format(f)] = data.groupby(['year','month', 'day', 'hour'])[f].transform('min')
    data['MDH_{}_std'.format(f)] = data.groupby(['year','month', 'day', 'hour'])[f].transform('std')
    data['MDH_{}_minmax'.format(f)]=data['MDH_{}_max'.format(f)]-data['MDH_{}_min'.format(f)]
    group_feats.append('MDH_{}_medi'.format(f))
    group_feats.append('MDH_{}_mean'.format(f))
    group_feats.append('MDH_{}_max'.format(f))
    group_feats.append('MDH_{}_min'.format(f))
    group_feats.append('MDH_{}_std'.format(f))
#交叉特征
for f1 in tqdm(['变频器电网侧有功功率', '外界温度','风向','dir_diff','dir_diff2'] + group_feats):

    for f2 in ['变频器电网侧有功功率', '外界温度','风向','dir_diff','dir_diff2'] + group_feats:
        if f1 != f2:
            colname = '{}_{}_ratio'.format(f1, f2)
            data[colname] = data[f1].values / data[f2].values

data = data.fillna(method='bfill')
dr_col=[]
for i in data.columns:
    if np.inf in data[i].unique():
        dr_col.append(i)
data=data.drop(dr_col,axis=1)
col=pd.DataFrame(data.columns)
if fc==1:
    name_col='col_dir.csv'
else:
    name_col='col_dir2.csv'
col.to_csv(name_col,index=False,encoding='gbk')
#离散化
'''
for f in fea:
    data[f + '_20_bin'] = pd.cut(data[f], 20, duplicates='drop').apply(lambda x: x.left).astype(float)
    data[f + '_50_bin'] = pd.cut(data[f], 50, duplicates='drop').apply(lambda x: x.left).astype(float)
    data[f + '_100_bin'] = pd.cut(data[f], 100, duplicates='drop').apply(lambda x: x.left).astype(float)
    data[f + '_200_bin'] = pd.cut(data[f], 200, duplicates='drop').apply(lambda x: x.left).astype(float)

for f1 in tqdm(
        [i+'_20_bin' for i in fea]):
    for f2 in fea:
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
#建模
def single_model(clf, train_x, train_y, test_x, clf_name, class_num=1):
    train = np.zeros((train_x.shape[0], class_num))
    test = np.zeros((test_x.shape[0], class_num))
    features=train_x.columns
    nums = int(train_x.shape[0] * 0.80)
    
    if clf_name in ['sgd', 'ridge']:
        print('MinMaxScaler...')

        ss = MinMaxScaler((0,1))
        ss.fit_transform(train_x.values)
        train_x = ss.transform(train_x.values)
        # test_x[col] = ss.transform(test_x[[col]].values)
        joblib.dump(ss,'mmx_dir.pkl')
    trn_x, trn_y, val_x, val_y = train_x[:nums], train_y[:nums], train_x[nums:], train_y[nums:]

    if clf_name == "lgb":
        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)
        data_matrix = clf.Dataset(train_x, label=train_y)

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
            'gpu_platform_id':1,
            'gpu_device_id':1
        }

        model = clf.train(params, train_matrix, 20000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,
                          early_stopping_rounds=1000,categorical_feature=['season'])
        model2 = clf.train(params, data_matrix, model.best_iteration,categorical_feature=['season'])
        val_pred = model.predict(val_x, num_iteration=model2.best_iteration).reshape(-1, 1)
        test_pred = model.predict(test_x, num_iteration=model2.best_iteration).reshape(-1, 1)
        if fc==1:
            name_lgb='lgbm_dir.txt'
        else:
            name_lgb='lgbm_dir2.txt'
        model2.save_model(name_lgb)
        
    if clf_name == "xgb":
        # xgb.XGBRegressor()
        train_matrix = clf.DMatrix(trn_x, label=trn_y, missing=np.nan)
        valid_matrix = clf.DMatrix(val_x, label=val_y, missing=np.nan)
        test_matrix = clf.DMatrix(test_x, missing=np.nan)
        params = {'booster': 'gbtree',
                  'eval_metric': 'mae',
                  'min_child_weight': 5,
                  'max_depth': 8,
                  'subsample': 0.5,
                  'colsample_bytree': 0.5,
                  'eta': 0.001,
                  'seed': 2020,
                  'nthread': 36,
                  'silent': 0,
                  'tree_method':'gpu_hist'
                  }

        watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

        model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500,
                          early_stopping_rounds=1000)
        val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
        test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit).reshape(-1, 1)

    if clf_name == "cat":
        params = {'learning_rate': 0.01, 'depth': 12, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False,
                  'task_type' : 'GPU'}

        model = clf(iterations=20000, **params)
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  cat_features=['season'], use_best_model=True, verbose=500)
        if fc==1:
            name_cat='cat_dir'
        else:
            name_cat='cat_dir2'
        model.save_model(name_cat)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)

    if clf_name == "sgd":
        params = {
            'loss': 'squared_loss',
            'penalty': 'l2',
            'alpha': 0.00001,
            'random_state': 2020,
        }
        model = SGDRegressor(**params)
        model.fit(trn_x, trn_y)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)
        joblib.dump(model,'sgd_dir.pkl')
        
    if clf_name == "ridge":
        params = {
            'alpha': 1.0,
            'random_state': 2020,
        }
        model = Ridge(**params)
        model.fit(trn_x, trn_y)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)
        joblib.dump(model,'ridge_dir.pkl')

    print("%s_mse_score:" % clf_name, mean_squared_error(val_y, val_pred))

    return val_pred, test_pred


def lgb_model(x_train, y_train, x_valid):
    lgb_train, lgb_test = single_model(lgb, x_train, y_train, x_valid, "lgb", 1)
    return lgb_train, lgb_test


def xgb_model(x_train, y_train, x_valid):
    xgb_train, xgb_test = single_model(xgb, x_train, y_train, x_valid, "xgb", 1)
    return xgb_train, xgb_test


def cat_model(x_train, y_train, x_valid):
    cat_train, cat_test = single_model(CatBoostRegressor, x_train, y_train, x_valid, "cat", 1)
    return cat_train, cat_test


def sgd_model(x_train, y_train, x_valid):
    sgd_train, sgd_test = single_model(SGDRegressor, x_train, y_train, x_valid, "sgd", 1)
    return sgd_train, sgd_test


def ridge_model(x_train, y_train, x_valid):
    ridge_train, ridge_test = single_model(Ridge, x_train, y_train, x_valid, "ridge", 1)
    return ridge_train, ridge_test

drop_columns = ["time", "month", "day",'hour','风速','y_test','year']

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

train_df = train_set.drop(drop_columns,axis=1).copy()
test_df = test_set.drop(drop_columns,axis=1).copy()

x_train = train_df.copy()
y_train = train_set['y_test'].copy()
x_test = test_df.copy()
y_test=test_set['y_test'].values

# lr_train, lr_test = ridge_model(x_train, y_train, x_test)

# sgd_train, sgd_test = sgd_model(x_train, y_train, x_test)

cat_train, cat_test = cat_model(x_train, y_train, x_test)

x_train['season']=x_train['season'].apply(lambda x:season_lgb(x)).astype('category')
x_test['season']=x_test['season'].apply(lambda x:season_lgb(x)).astype('category')
# lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)

error_cat=mean_squared_error(cat_test,y_test)
# error_lgb=mean_squared_error(lgb_test,y_test)
print('err_cat:',error_cat)
# print('err_lgb:',error_lgb)
# xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)


# y_pre=(lr_test+sgd_test+lgb_test+xgb_test+cat_test)/5
# err=mean_squared_error(y_test,y_pre)
# train_matrix = xgb.DMatrix(x_train, label=y_train, missing=np.nan)
# plt.plot(lgb_test,c='g')
# plt.plot(y_test,c='r')
# plt.show()