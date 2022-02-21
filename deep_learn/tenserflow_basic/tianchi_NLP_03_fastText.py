# -*- coding: utf-8 -*-
"""
@Time     :2022/2/21 15:39
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

"""
不同于机器学习，深度学习即能够进行特征提取，也可以完成分类功能。
现有文本表示方法的缺陷：
    one hot/ bag of word/N-gram/TF-IDF;
    转换得到的向量维度很高，训练耗时长，仅根据统计没有考虑单词之间的关系。
深度学习中的文本表示方法：
    FastText/word2vec/bert，他们可以进行文本表示，同时能将其映射到一个低纬度空间中；
    
FastText
    是一种典型的深度学习词向量表示方法，通过Embedding层将单词映射到稠密空间，然后将句子中所有单词在Embedding空间进行平均，进而完成分类操作；
    fastText是三层神经网络，input layer;hidden layer;output layer；
    文本分类中fastText由于TFIDF:
        fast用单词的embedding叠加获取文档向量，将相似的句子分为一类；
        fasttext学习得到的embedding空间维度比较低，训练速度块。
    FastText可以快速的在CPU上进行训练，最好的实践方法就是官方开源的版本： https://github.com/facebookresearch/fastText/tree/master/python

"""
##-- FastText网络结构实现简介
# from __future__ import unicode_literals
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense
VOCAB_SIZE = 2000   # 词典大小
EMBEDDING_DIM = 100 # 词嵌入维度
MAX_WORDS = 500     # 句子最大长度
CLASS_NUM = 5
def build_fastText():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_WORDS))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(CLASS_NUM, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=['acc'])
    return model

def flod_10(total,all_labels):
    label2id={}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

if __name__ == "__main__":
    model = build_fastText()
    print(model.summary())