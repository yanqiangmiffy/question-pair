# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: gen_fea_fuzz.py 
@Time: 2018/12/6 15:56
@Software: PyCharm 
@Description:
"""
from fuzzywuzzy import fuzz
import pandas as pd

df_train = pd.read_csv('../data/input/atec_nlp_sim_train.csv',
                       encoding='utf-8-sig', header=None, sep='\t')
df_train.columns = ['line', 'q1', 'q2', 'label']

df_train_add = pd.read_csv('../data/input/atec_nlp_sim_train_add.csv',
                           encoding='utf-8-sig', header=None, sep='\t')
df_train_add.columns = ['line', 'q1', 'q2', 'label']
df_train = pd.concat([df_train, df_train_add], axis=0, sort=False)

df_feat = pd.DataFrame()
df_feat['fuzz_ratio']=df_train.apply(lambda row:fuzz.ratio(str(row.q1),str(row.q2)),axis=1)
df_feat['fuzz_qratio']=df_train.apply(lambda row:fuzz.QRatio(str(row.q1),str(row.q2)),axis=1)
df_feat['fuzz_wratio']=df_train.apply(lambda row:fuzz.WRatio(str(row.q1),str(row.q2)),axis=1)
df_feat['fuzz_partial_ratio']=df_train.apply(lambda row:fuzz.partial_ratio(str(row.q1),str(row.q2)),axis=1)
df_feat['fuzz_partial_token_set_ratio']=df_train.apply(lambda row:fuzz.partial_token_set_ratio(str(row.q1),str(row.q2)),axis=1)
df_feat['fuzz_partial_token_sort_ratio']=df_train.apply(lambda row:fuzz.partial_token_sort_ratio(str(row.q1),str(row.q2)),axis=1)
df_feat['fuzz_token_set_ratio']=df_train.apply(lambda row:fuzz.token_set_ratio(str(row.q1),str(row.q2)),axis=1)
df_feat['fuzz_token_sort_ratio']=df_train.apply(lambda row:fuzz.token_sort_ratio(str(row.q1),str(row.q2)),axis=1)

df_feat.to_csv('subfeas/train_feature_fuzz.csv',index=False)