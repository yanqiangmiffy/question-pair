# question-pair
Duplicated  Question Pairs

## 1 特征工程
### 1.1 NLP特征
文本长度特征
- 字符长度，长度差,比例
- 单词(分词后)长度，长度差,比例

difflib
- 字符比较特征

字符串模糊匹配 fuzz
- 简单比 fuzz.ratio
- 部分比 fuzz.partial_ratio
- 单词集合比 fuzz.token_set_ratio
- 单词排序比 fuzz.token_sort_ratio
-  fuzz.QRatio
-  fuzz.WRatio
-  fuzz.partial_token_set_ratio
-  fuzz.partial_token_sort_ratio

Tfidf

- tfidf_sum tfidf相加之后的和
- tfidf_mean tfidf的平均值
- tfidf_len tfidf向量的长度

simhash

- simhash_tokenize_distance
- simhash_word_2gram_distance
- simhash_word_3gram_distance
- simhash_char_3gram_distance

embedding 特征

主要利用scipy计算两个句子向量中的各种度量，
> 这里为了省事直接使用了doc2vec，可以快速标示句子向量

- skew
> 偏度（skewness），是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。
偏度(Skewness)亦称偏态、偏态系数。 
- kurtosis

峰度是描述总体中所有取值分布形态陡缓程度的统计量。

句子相似度度量
- doc_sims
- cosine 余弦距离
- cityblock 曼哈顿距离
- jaccard
 > 杰卡德距离(Jaccard Distance) 是用来衡量两个集合差异性的一种指标，它是杰卡德相似系数的补集，被定义为1减去Jaccard相似系数
- canberra 堪培拉距离

![](https://people.revoledu.com/kardi/tutorial/Similarity/image/CanberraDistance_clip_image004.gif)

- euclidean 欧氏距离
- minkowski 
> 明氏距离又叫做明可夫斯基距离，是欧氏空间中的一种测度，被看做是欧氏距离和曼哈顿距离的一种推广

![](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D122/sign=d199862ca764034f0bcdc6049dc27980/1c950a7b02087bf490b31f18fed3572c10dfcfe0.jpg)
- braycurtis

> Bray-Curtis距离是以该统计指标的提出者J. Roger Bray和John T. Curtis的名字命名的，主要基于OTUs的计数统计，
比较两个群落微生物的组成差异。与unifrac距离，包含的信息完全不一样；相比于jaccard距离，Bray-Curtis则包含了OTUs丰度信息。
![](http://www.dengfeilong.com/uploads/allimg/180111/1-1P1110645025Q.png)

## 参考资料
- [3.1.6 峰度（Kurtosis）和偏度（Skewness）](https://blog.csdn.net/binbigdata/article/details/79897160)
- [微生物多样性分析之Bray curtis距离](http://www.dengfeilong.com/weishengwu/130.html)