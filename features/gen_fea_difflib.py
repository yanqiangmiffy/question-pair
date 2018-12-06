# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: gen_fea_difflib.py
@Time: 2018/12/6 15:38
@Software: PyCharm 
@Description:
"""
# ratio（）¶
# 返回一个值，该值测量序列与[0,1]范围内浮点数的相似性。
# 假设T是两个序列中元素数的总和，设M是匹配的数，该值表示为2.0 * M / T. 如果序列完全相同，则值将是1.0，如果它完全不同0.0：

import difflib
import pandas as pd
import numpy as np

df_train = pd.read_csv('../data/input/atec_nlp_sim_train.csv',
                       encoding='utf-8-sig', header=None, sep='\t')
df_train.columns = ['line', 'q1', 'q2', 'label']

df_train_add = pd.read_csv('../data/input/atec_nlp_sim_train_add.csv',
                           encoding='utf-8-sig', header=None, sep='\t')
df_train_add.columns = ['line', 'q1', 'q2', 'label']
df_train = pd.concat([df_train, df_train_add], axis=0, sort=False)

df_feat = pd.DataFrame()


def diff_ratios(row):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(row.q1), str(row.q2))
    return seq.ratio()


df_feat['diff_ratios'] = df_train.apply(
    lambda row: diff_ratios(row), axis=1)
df_feat.to_csv('subfeas/train_feature_difflib.csv', index=False)
