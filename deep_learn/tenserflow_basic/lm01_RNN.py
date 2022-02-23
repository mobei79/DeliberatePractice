# -*- coding: utf-8 -*-
"""
@Time     :2022/2/16 15:09
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
"""
https://blog.csdn.net/qq_38574975/article/details/107528825 基于tensorflow2.0的RNN 情感分析

"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
import io
import os
import pandas as pd

tf.random.set_seed(22)
np.random.seed(22)
# os.environ["jjjjj"] = 2
assert tf.__version__.startswith('2.')

batchsz = 128

total_words = 10000 # the most frequest words 频率最高的10000份单词，也就是字典容量
max_review_len = 80 # 序列长度
embedding_len = 100 # 词嵌入长度

# x_train :[25000, ] 样本包含25000行，每行单词数目不一定
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
"""# Set = set()
# for i in x_train:
#     Set.update(np.unique(i))
# print(Set)
# print(len(Set))
# print(len(x_train[0]))
# print(len(x_train[3]))
# print(x_train.shape)
# print(x_test.shape)"""

# x_train:[25000, 80]
# x_text:[25000, 80]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
print(x_train.shape)


db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print("x_train shape:",x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print("x_test shape:" ,x_test.shape)
# x_train shape: (25000, 80) tf.Tensor(1, shape=(), dtype=int64) tf.Tensor(0, shape=(), dtype=int64)
# x_test shape: (25000, 80)

class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len,
                                                   input_length=max_review_len)
        # [b, 80, 100] ,h_dim :64
        self.rnn = keras.Sequential([
            tf.keras.layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            tf.keras.layers.SimpleRNN(units, dropout=0.5, unroll=True)
        ])

        # fc, [b, 80, 100] => [b,1]
        self.outlayer = tf.keras.layers.Dense(1)
    def call(self, inputs, training=None, mask=None):
        #[b, 80]
        x = inputs
        # embedding [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # run cell compute x:[b, 80, 100] => [b, 64]
        x = self.rnn(x)
        # out:[b, 64] =>[b,1]
        s = self.outlayer(x)
        # p(y is pos|x)
        prop = tf.sigmoid(x)
        return prop

def main():
    units = 64
    epochs = 4
    model = MyRNN(units)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs = epochs, validation_data=db_test)

    """
    
    """
    model.evaluate(db_test)


if __name__ == "__main__":
    main()


# # from tensorflow import keras
# print(tf.__version__)
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import keras
# import json
# import tensorflow_datasets as tfds
#
# # 加载数据 这里使用的是subword8k版本，也就是子词数据集
# imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
# train_data, test_data = imdb['train'], imdb["test"]
# tokenizer = info.features['text'].encoder  # 通过这个语句获得一个子词的分词器，这是一个预训练好的子词分类器。【这里的数据是已经分词后的】
# # print(tokenizer.subwords)  # 查看分词器的词汇表
# print(tokenizer.vocab_size)
# # 查看她如何对字符串进行编码的
# # sample_string = "TensorFlow, from basics to mastery"
# # tokenized_string = tokenizer.encode(sample_string)  # 编码
# # print(tokenized_string)
# # original_string = tokenizer.decode(tokenized_string)  # 解码
# # print(original_string)
# # for ts in tokenized_string:  # 查看编码具体内容
# #     print("{}------>{}".format(ts, tokenizer.decode([ts])))
#
#
# BUFFER_SIZE = 1000  # 25000
# BATCH_SIZE = 1
# train_data = train_data.shuffle(BUFFER_SIZE)
# train_data = train_data.padded_batch(BATCH_SIZE)
# test_data = test_data.padded_batch(BATCH_SIZE)
# """
# 构建神经网络模型
# """
# # embedding_dim = 64
# # model = tf.keras.Sequential([
# #     tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),  # 嵌入层的结果是一个二维的数组，句子长度*向量维度
# #     # keras.layers.Flatten(),  # 平坦层展平
# #     keras.layers.GlobalAveragePooling1D(),  # 全局平均池化层， 在每个向量的维度上取平均值输出，得到模型更加简洁，速度更快;有些编码也难以展平
# #     keras.layers.Dense(6, activation="relu"),
# #     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# # ])
# # model.summary()
#
# """RNN
#     Embedding函数的三个参数： 本质上是对输入数据降维的过程；
#         input_dim: 指输入输入数据的维度【字典的维度】，进行embedding时每个单词都是input_dim维度的onehot向量。
#         output_dim:指输出数据的维度，也就是经过embedding层降维后的数据是由多少元素组成
#         input_length:该参数指定输入数据的长度，相当于在Embedding层之前加了一个Input层。在链接Flastten或者Dense层之前，需要这样一个input长度，因为Dense层是全连接层，所有的长度需要指定；
#                     意思就是输入数据的从长度是多少（数据是由多少个重复的元素按一定的顺序组合而成的）
#     经过embedding层处理后，数据会增加一个维度，新增的维度为onehot编码，表名当前数据在数据列表中的索引。
# 8185
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 80, 100)           818500
# _________________________________________________________________
# simple_rnn (SimpleRNN)       (None, 80, 64)            10560
# _________________________________________________________________
# simple_rnn_1 (SimpleRNN)     (None, 64)                8256
# _________________________________________________________________
# dense (Dense)                (None, 64)                4160
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 65
# =================================================================
# Total params: 841,541
# Trainable params: 841,541
# Non-trainable params: 0
# """
# embedding_dim = 64
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(tokenizer.vocab_size, 100, input_length=80),  # 输出80（句子长度） * 100（embedding维度）；参数：字典长度*embedding维度，即8185*100
#     tf.keras.layers.SimpleRNN(64, return_sequences=True, unroll=True),  #
#     # tf.keras.layers.SimpleRNN(64, unroll=True),
#     # tf.keras.layers.GlobalAveragePooling1D(),
#     # tf.keras.layers.Dense(64, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
# model.summary()
#
