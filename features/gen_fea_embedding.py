# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: gen_fea_embedding.py 
@Time: 2018/12/7 10:53
@Software: PyCharm 
@Description:
"""
import pandas as pd
import numpy as np
import jieba
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from tqdm import tqdm

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



df_train['qid1'] = ['qid1_' + str(i) for i in range(len(df_train.q1))]
df_train['qid2'] = ['qid2_' + str(i) for i in range(len(df_train.q2))]
qids = df_train.qid1.tolist() + df_train.qid2.tolist()


def train_doc2vec():
    print("training doc2vec...")
    df_train.q1 = df_train.q1.apply(lambda x: tokenize(x))
    df_train.q2 = df_train.q2.apply(lambda x: tokenize(x))
    questions = df_train.q1.tolist() + df_train.q2.tolist()

    tag_tokenized = [TaggedDocument(question, [id])
                     for question, id in zip(questions, qids)]
    model = Doc2Vec(size=300, min_count=1, epochs=200)
    model.build_vocab(tag_tokenized)
    model.train(tag_tokenized, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('tmp/word_doc2vec.model')


# train_doc2vec()
doc_sims = []
cos_distance = []
cityblock_distance = []
jaccard_distance = []
canberra_distance = []
euclidean_distance = []
minkowski_distance = []
braycurtis_distance = []
skew_q1vec = []
skew_q2vec = []
kur_q1vec = []
kur_q2vec = []
model_dm = Doc2Vec.load('tmp/word_doc2vec.model')

for qid1, qid2 in tqdm(zip(df_train['qid1'], df_train['qid2'])):
    doc_sims.append(model_dm.docvecs.similarity(qid1, qid2))  # 参数文档对应的id
    cos_distance.append(cosine(model_dm.docvecs[qid1], model_dm.docvecs[qid2]))
    cityblock_distance.append(cityblock(model_dm.docvecs[qid1], model_dm.docvecs[qid2]))
    jaccard_distance.append(jaccard(model_dm.docvecs[qid1], model_dm.docvecs[qid2]))
    canberra_distance.append(canberra(model_dm.docvecs[qid1], model_dm.docvecs[qid2]))
    euclidean_distance.append(euclidean(model_dm.docvecs[qid1], model_dm.docvecs[qid2]))
    minkowski_distance.append(minkowski(model_dm.docvecs[qid1], model_dm.docvecs[qid2]))
    braycurtis_distance.append(braycurtis(model_dm.docvecs[qid1], model_dm.docvecs[qid2]))
    skew_q1vec.append(skew(model_dm.docvecs[qid1]))
    skew_q2vec.append(skew(model_dm.docvecs[qid2]))
    kur_q1vec.append(skew(model_dm.docvecs[qid1]))
    kur_q2vec.append(skew(model_dm.docvecs[qid2]))

df_feat['doc_sim'] = doc_sims
df_feat['cos_distance'] = cos_distance
df_feat['cityblock_distance'] = cityblock_distance
# df_feat['jaccard_distance'] = jaccard_distance
df_feat['canberra_distance'] = canberra_distance
df_feat['euclidean_distance'] = euclidean_distance
df_feat['minkowski_distance'] = minkowski_distance
df_feat['braycurtis_distance'] = braycurtis_distance
df_feat['skew_q1vec'] = skew_q1vec
df_feat['skew_q2vec'] = skew_q2vec
df_feat['kur_q1vec'] = kur_q1vec
df_feat['kur_q2vec'] = kur_q2vec


df_feat.to_csv('subfeas/train_feature_embedding.csv',index=False)
