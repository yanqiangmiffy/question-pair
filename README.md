# question-pair
Duplicated  Question Pairs

## 1 特征工程
### 1.1 NLP特征
文本长度特征
- 字符长度，长度差,比例
- 单词(分词后)长度，长度差,比例

difflib
- 字符比较特征

字符串模糊匹配
- 简单比 fuzz.ratio
- 部分比 fuzz.partial_ratio
- 单词集合比 fuzz.token_set_ratio
- 单词排序比 fuzz.token_sort_ratio
-  fuzz.QRatio
-  fuzz.WRatio
-  fuzz.partial_token_set_ratio
-  fuzz.partial_token_sort_ratio
