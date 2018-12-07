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
- canberra
- euclidean 
- minkowski
- braycurtis


## 参考资料
- [3.1.6 峰度（Kurtosis）和偏度（Skewness）](https://blog.csdn.net/binbigdata/article/details/79897160)