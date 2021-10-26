# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:56:32 2021

@author: liuzhenguo
"""

import pandas as pd

data_fs=pd.read_csv('result_风速.csv')
data_fs1=pd.read_csv('result_风速1.csv')
data_fx=pd.read_csv('result_风向.csv')
data_fx1=pd.read_csv('result_风向1.csv')

sub=pd.DataFrame()
fc1=['风场1' for i in range(40000)]
fc2=['风场2' for i in range(39220)]
fc=fc1+fc2
fc=pd.DataFrame({'风场':fc})
sd=pd.concat([data_fs['时段'],data_fs1['时段']],axis=0,ignore_index=True)
fj=pd.concat([data_fs['风机'],data_fs1['风机']],axis=0,ignore_index=True)
sk=pd.concat([data_fs['时刻'],data_fs1['时刻']],axis=0,ignore_index=True)
fs=pd.concat([data_fs['风速'],data_fs1['风速']],axis=0,ignore_index=True)
fx=pd.concat([data_fx['风向'],data_fx1['风向']],axis=0,ignore_index=True)
sub=pd.concat([fc,fj,sd,sk,fs,fx],axis=1)
sub.to_csv('sub_2021-10-25.csv',index=False)
'''
#风场1
#春
sub=pd.DataFrame()
for i in range(1,21):
    if i <10 :
        a='春_0'+str(i)
    else:
       a='春_'+str(i) 
    data_tem=data_fs.query('时段 == @a')
    data_tem1=data_fx.query('时段 == @a')
    data_tem['风向']=data_tem1['风向']
    sub=pd.concat([sub,data_tem],ignore_index=True)
#夏天
for i in range(1,21):
    if i <10 :
        a='夏_0'+str(i)
    else:
       a='夏_'+str(i) 
    data_tem=data_fs.query('时段 == @a')
    data_tem1=data_fx.query('时段 == @a')
    data_tem['风向']=data_tem1['风向']
    sub=pd.concat([sub,data_tem],ignore_index=True)
#秋
for i in range(1,21):
    if i <10 :
        a='秋_0'+str(i)
    else:
       a='秋_'+str(i) 
    data_tem=data_fs.query('时段 == @a')
    data_tem1=data_fx.query('时段 == @a')
    data_tem['风向']=data_tem1['风向']
    sub=pd.concat([sub,data_tem],ignore_index=True)
#冬
for i in range(1,21):
    if i <10 :
        a='冬_0'+str(i)
    else:
       a='冬_'+str(i) 
    data_tem=data_fs.query('时段 == @a')
    data_tem1=data_fx.query('时段 == @a')
    data_tem['风向']=data_tem1['风向']
    sub=pd.concat([sub,data_tem],ignore_index=True)
#风场2
#春
for i in range(1,21):
    if i <10 :
        a='春_0'+str(i)
    else:
       a='春_'+str(i) 
    data_tem=data_fs1.query('时段 == @a')
    data_tem1=data_fx1.query('时段 == @a')
    data_tem['风向']=data_tem1['风向']
    sub=pd.concat([sub,data_tem],ignore_index=True)
#夏
for i in range(1,21):
    if i <10 :
        a='夏_0'+str(i)
    else:
       a='夏_'+str(i) 
    data_tem=data_fs1.query('时段 == @a')
    data_tem1=data_fx1.query('时段 == @a')
    data_tem['风向']=data_tem1['风向']
    sub=pd.concat([sub,data_tem],ignore_index=True)
#秋
for i in range(1,21):
    if i <10 :
        a='秋_0'+str(i)
    else:
       a='秋_'+str(i) 
    data_tem=data_fs.query('时段 == @a')
    data_tem1=data_fx.query('时段 == @a')
    data_tem['风向']=data_tem1['风向']
    sub=pd.concat([sub,data_tem],ignore_index=True)
#冬
for i in range(1,21):
    if i <10 :
        a='冬_0'+str(i)
    else:
       a='冬_'+str(i) 
    data_tem=data_fs.query('时段 == @a')
    data_tem1=data_fx.query('时段 == @a')
    data_tem['风向']=data_tem1['风向']
    sub=pd.concat([sub,data_tem],ignore_index=True)

'''