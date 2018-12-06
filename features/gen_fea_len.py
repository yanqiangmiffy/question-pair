# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: gen_fea_len.py
@Time: 2018/12/6 14:43
@Software: PyCharm 
@Description:
"""
import jieba
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

# 字符长度
df_feat['char_len1'] = df_train.q1.map(lambda x: len(str(x)))
df_feat['char_len2'] = df_train.q2.map(lambda x: len(str(x)))


# 单词词长度
def get_word_len(x):
    return len([word for word in jieba.cut(str(x))])


df_feat['word_len1'] = df_train.q1.map(lambda x: get_word_len(x))
df_feat['word_len2'] = df_train.q2.map(lambda x: get_word_len(x))

# 长度差
df_feat['char_len_diff']=df_feat.apply(
    lambda row:abs(row.char_len1-row.char_len2),axis=1) # axis=1 每行
df_feat['word_len_diff']=df_feat.apply(
    lambda row:abs(row.word_len1-row.word_len2),axis=1)

# 比例
df_feat['char_len_diff_ratio'] = df_feat.apply(
    lambda row: abs(row.char_len1 - row.char_len2) / (row.char_len1 + row.char_len2), axis=1)
df_feat['word_len_diff_ratio'] = df_feat.apply(
    lambda row: abs(row.word_len1 - row.word_len2) / (row.word_len1 + row.word_len2), axis=1)
df_feat.to_csv('subfeas/train_feature_len.csv', index=False)
