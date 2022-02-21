# -*- coding: utf-8 -*-
"""
@Time     :2022/2/21 14:15
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
"""
使用机器学习的方法解决文本分类问题
    TF-IDF
    sklearn机器学习模型
    
    文本表示方法：
        假设N个样本，M个特征，组成N*M维的样本特征矩阵；图片可以看做high*weight*3的特征图矩阵；
        在自然语言领域：文本长度不定，将文本表示成机器能够运算的数字或者向量的方法称为词嵌入（word embedding）方法，词嵌入将不定长文本转换到定长的空间中，这时文本分类的第一步
    one hot表示
        即将单词用一个离散向量表示，具体的将每个字词编码一个索引，然后根据索引进行复制；
        eg.
            我：[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            爱：[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ...
            海：[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    bag of word 词袋模型（也称count vectors）
        eg.
            句子1：我 爱 北 京 天 安 门
            转换为 [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
            句子2：我 喜 欢 上 海
            转换为 [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        每个文档的字词可以使用期出现频率进行表示；
        sklearn中的CountVectorizer可以实现这一步骤。
        如keras中的Tokenizer fit_on_texts方法。将文字进行编码。然后使用texts_to_sequences进行序列化
    N-gram
        N-gram和count vectors类似，不过是加入了相邻单词组合成为新的单词，并进行计数。
        如果N取2，如下:
        eg.
            句子1：我爱 爱北 北京 京天 天安 安门
            句子2：我喜 喜欢 欢上 上海
    TF-IDF
        由两部分组成，第一部分：词语频率(term frequency)；第二部分逆文档频率（inverse document frequency）
        语料库中文档总数除以含有该词语的文档数目，再取对数就是逆文档频率。
            TF = 该词汇会当前文档中出现次数/ 当前文档中词语总数 
            IDF = log_e(文档总数/出现该词语的文档总数)     
"""
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

tianchi_data_path = "D:\Program Files\pythonwork\dataset\DeliberatePractice\deep_learn\\tianchi\\"
train_df = pd.read_csv(tianchi_data_path+"train_set.csv",
                       sep="\t",        # csv文件每列的分割符号 \t
                       nrows=15000)       # 读取行数

"""
机器学习 逻辑回归
"""
# vectorizer = CountVectorizer(max_features=3000)
# train_test = vectorizer.fit_transform(train_df['text'])
#
# clf = RidgeClassifier()
# clf.fit(train_test[:10000], train_df['label'].values[:10000]) # Series.values 以内置ndarray-like形式返回
#
# val_pred = clf.predict(train_test[10000:])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))





"""
# TF-IDF +  RidgeClassifier
"""
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))


"""
词袋模型
"""
# from sklearn.feature_extraction.text import CountVectorizer
# sentences = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
# vectorizer = CountVectorizer()
# sequences = vectorizer.fit_transform(sentences).toarray()
# print(sequences)