# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: gen_fea_simhash.py 
@Time: 2018/12/6 17:42
@Software: PyCharm 
@Description:
"""
import pandas as pd
import numpy as np
import jieba
from nltk import ngrams
from simhash import Simhash

df_train = pd.read_csv('../data/input/atec_nlp_sim_train.csv',
                       encoding='utf-8-sig', header=None, sep='\t')
df_train.columns = ['line', 'q1', 'q2', 'label']
df_train_add = pd.read_csv('../data/input/atec_nlp_sim_train_add.csv',
                           encoding='utf-8-sig', header=None, sep='\t')
df_train_add.columns = ['line', 'q1', 'q2', 'label']
df_train = pd.concat([df_train, df_train_add], axis=0, sort=False)
df_feat = pd.DataFrame()


def tokenize(sequence):
    return [word for word in jieba.cut(sequence)]


def clean_sequence(sequence):
    tokens = tokenize(sequence)
    return ' '.join(tokens)


def get_word_ngrams(sequence, n=3):
    tokens = tokenize(sequence)
    return [' '.join(ngram) for ngram in ngrams(tokens, n)]


def get_character_ngrams(sequence, n=3):
    sequence = clean_sequence(sequence)
    return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]


def caluclate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))


def get_word_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return caluclate_simhash_distance(q1, q2)


def get_word_2gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)


def get_char_2gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)


def get_word_3gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)


def get_char_3gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)

df_train['questions'] = df_train['q1'] + '_split_tag_' + df_train['q2']


df_feat['simhash_tokenize_distance'] = df_train['questions'].apply(get_word_distance)
df_feat['simhash_word_2gram_distance'] = df_train['questions'].apply(get_word_2gram_distance)
df_feat['simhash_char_2gram_distance'] = df_train['questions'].apply(get_char_2gram_distance)
df_feat['simhash_word_3gram_distance'] = df_train['questions'].apply(get_word_3gram_distance)
df_feat['simhash_char_3gram_distance'] =df_train['questions'].apply(get_char_3gram_distance)


df_feat.to_csv('subfeas/train_feature_simhash.csv',index=False)