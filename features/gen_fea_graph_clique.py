# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: gen_fea_graph_clique.py 
@Time: 2018/12/6 16:05
@Software: PyCharm
@Description:团（clique）是图论中的用语。
"""
"""
团（clique）是图论中的用语。对于给定图G=(V,E)。
其中，V={1,…,n}是图G的顶点集，E是图G的边集。
图G的团就是一个两两之间有边的顶点集合。简单地说，团是G的一个完全子图。
如果一个团不被其他任一团所包含，即它不是其他任一团的真子集，则称该团为图G的极大团（maximal clique）。顶点最多的极大团，称之为图G的最大团（maximum clique）。
最大团问题的目标就是要找到给定图的最大团。 [1] 
"""

import networkx as nx
import pandas as pd
from itertools import combinations

df_train = pd.read_csv('../data/input/atec_nlp_sim_train.csv',
                       encoding='utf-8-sig', header=None, sep='\t')
df_train.columns = ['line', 'q1', 'q2', 'label']

df_train_add = pd.read_csv('../data/input/atec_nlp_sim_train_add.csv',
                           encoding='utf-8-sig', header=None, sep='\t')
df_train_add.columns = ['line', 'q1', 'q2', 'label']
df_train = pd.concat([df_train, df_train_add], axis=0, sort=False)
df_feat = pd.DataFrame()

G = nx.Graph()
edges = [tuple(x) for x in df_train[['q1', 'q2']].values]
G.add_edges_from(edges) # q1->q2 形成一条边

map_label = dict(((x[0], x[1])) for x in df_train[['q1', 'q2']].values)

map_clique_size = {}
cliques = sorted(list(nx.find_cliques(G)), key=lambda x: len(x))
for cli in cliques:
    for q1, q2 in combinations(cli, 2):
        if (q1, q2) in map_label:
            map_clique_size[q1, q2] = len(cli)
        elif (q2, q1) in map_label:
            map_clique_size[q2, q1] = len(cli)

df_feat['clique_size'] = df_train.apply(lambda row: map_clique_size.get((row['q1'], row['q2']), -1), axis=1)

df_feat.to_csv('subfeas/train_feature_clique.csv',index=False)

# 问题没有重复的，此特征不需要