# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: gen_fea_tfidf.py 
@Time: 2018/12/6 17:11
@Software: PyCharm 
@Description:
"""
import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df_train = pd.read_csv('../data/input/atec_nlp_sim_train.csv',
                       encoding='utf-8-sig', header=None, sep='\t')
df_train.columns = ['line', 'q1', 'q2', 'label']

df_train_add = pd.read_csv('../data/input/atec_nlp_sim_train_add.csv',
                           encoding='utf-8-sig', header=None, sep='\t')
df_train_add.columns = ['line', 'q1', 'q2', 'label']
df_train = pd.concat([df_train, df_train_add], axis=0, sort=False)
df_feat = pd.DataFrame()


def cut_sent(x):
    return " ".join(jieba.cut(str(x)))


df_train.q1 = df_train.q1.apply(lambda x: cut_sent(x))
df_train.q2 = df_train.q1.apply(lambda x: cut_sent(x))
corpus = pd.Series(df_train.q1.tolist() + df_train.q2.tolist()).astype(str)

tfidf=TfidfVectorizer(ngram_range=(1, 1))
tfidf.fit_transform(corpus)

tfidf_sum1 = []
tfidf_sum2 = []
tfidf_mean1 = []
tfidf_mean2 = []
tfidf_len1= []
tfidf_len2 = []

for index,row in df_train.iterrows():
    tfidf_q1=tfidf.transform([row.q1]).data
    tfidf_q2=tfidf.transform([row.q2]).data

    tfidf_sum1.append(np.sum(tfidf_q1))
    tfidf_sum2.append(np.sum(tfidf_q2))

    tfidf_mean1.append(np.mean(tfidf_q1))
    tfidf_mean2.append(np.mean(tfidf_q2))

    tfidf_len1.append(len(tfidf_q1))
    tfidf_len2.append(len(tfidf_q2))

df_feat['tfidf_sum1'] = tfidf_sum1
df_feat['tfidf_sum2'] = tfidf_sum2
df_feat['tfidf_mean1'] = tfidf_mean1
df_feat['tfidf_mean2'] = tfidf_mean2
df_feat['tfidf_len1'] = tfidf_len1
df_feat['tfidf_len2'] = tfidf_len2

df_feat.fillna(0.0)

df_feat.to_csv('subfeas/train_feature_tfidf.csv',index=False)
