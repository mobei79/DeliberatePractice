# # -*- coding: utf-8 -*-
# """
# @Time     :2022/2/15 11:03
# @Author   :jingjin.guo@fengjr.com
# @Last Modified by:
# @Last Modified time:
# @file :
# @desc :
# """
#
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
#
# """
# ##分词器 - 词条化 = 构建词典+编码
#
# tokenizer (塔克奈泽) 是分词器 快速产生词典，并创建词向量
#     标点符号对单词没有影响；
#     大小写不敏感；
#     能够将文本转换为tokens流；将单词转化位数字，将神经网络训练变为可能（神经网络只能进行数学计算）
# """
# sentences = [
#     "I love my dog .",
#     "I love my cat ,",
#     "you love my dog!",
#     "Do you think my dog is amazing?"
# ]
# tokenizer = Tokenizer(num_words=100, oov_token="<OOV")  # 表示建立一个100个词的词典，分词器选择将词频最高的100个单词放入词典，并编码。
# # oov_token未登录词
# # 词频低的单词对神经网络训练精度影响较小，但是会极大增加训练时间，需要仔细设置该参数
# tokenizer.fit_on_texts(sentences)  # 按照词频顺序编码
# rst = tokenizer.word_index  # 键值对
# print(rst)
#
# """
# 序列化
#     是将词条化的编码，基于编码将句子序列化，进一步对序列化的列表进行处理，使得每个句子的编码长度相同，才能训练神经网络。
#
#
#     相同的词典将用在神经网络的训练和推断两个阶段;
#      会丢失不在词典中的单词；
#         我们需要大量的数据来扩充我们的词典，否则就会出现缺失单词导致语句不通顺的情况；遇到未登录词是输入一个特殊值代替oov_token(唯一的)可以保存句子的结构；
#         预料越多，词典中的单词也越多
# """
#
# sequences = tokenizer.texts_to_sequences(sentences)
# print(sequences)  # 编码为整数的sentences数组
# test_data = [
#     "i really love my dog",
#     "my dog loves my manatee"
# ]
# test_seq = tokenizer.texts_to_sequences(test_data)  # 会丢失不在词典中的单词
# print(test_seq)
#
# """
# # 编码后的句子进行补齐
#     # 构建神经网络输入需要维度的一致性
#     分词器创建了序列，这些序列就可以被传递到pad_sequences中，以便进行补齐
#     maxlen控制句子长度，会丢失前面的信息，可以使用trucating=‘post’控制
# """
# padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)  # 默认是补充在前面,
# print(padded)