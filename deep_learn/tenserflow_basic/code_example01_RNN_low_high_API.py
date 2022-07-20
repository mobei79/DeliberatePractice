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
# 设置相关底层配置
# phtsical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(phtsical_devices) > 0 , "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(phtsical_devices[0], True)

batchsz = 128
total_words = 10000 # the most frequest words 频率最高的10000份单词，也就是字典容量,超出后按照未登录词处理；
max_review_len = 80 # 序列长度
embedding_len = 100 # 词嵌入长度

# x_train :[25000, ] 样本包含25000行，每行单词数目不一定
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
"""
# Set = set()
# for i in x_train:
#     Set.update(np.unique(i))
# print(Set)
# print(len(Set))
# print(len(x_train[0]))
# print(len(x_train[3]))
# print(x_train.shape)
# print(x_test.shape)
"""
print("load_data - x_train shape", x_train.shape, type(x_train))
print("load_data - y_train shape", y_train.shape, type(y_train))
print("load_data - x_train example", x_train[1])
print("load_data - y_train example", y_train[1])
# x_train:[25000, 80]
# x_text:[25000, 80]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
print("pad_sequences - x_train shape", x_train.shape, type(x_train))
print("pad_sequences - y_train shape", y_train.shape, type(y_train))
print("pad_sequences - x_train example", x_train[1])
print("pad_sequences - y_train example", y_train[1])


db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size=batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batch_size=batchsz, drop_remainder=True)

print("x_train shape:",x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print("x_test shape:" ,x_test.shape)
# x_train shape: (25000, 80) tf.Tensor(1, shape=(), dtype=int64) tf.Tensor(0, shape=(), dtype=int64)
# x_test shape: (25000, 80)

"""
上面为导入数据：样本25000；长度为80的句子
"""
class MyRNNLowAPI(keras.Model):

    def __init__(self, unit):
        super(MyRNNLowAPI, self).__init__()

        # 初始化参数 [b, 64]
        self.state0 = [tf.zeros([batchsz, unit])]
        self.state1 = [tf.zeros([batchsz, unit])]

        # [b, 80, 100]
        self.embedding = keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        #[b, 80, 100]  h_dim = 64
        # RNN: cell1, cell2,cell3
        # sampleRNN
        self.rnn_call0 = keras.layers.SimpleRNNCell(unit, dropout=0.5)
        self.rnn_call1 = keras.layers.SimpleRNNCell(unit, dropout=0.5)

        # fc: [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = keras.layers.Dense(1)

    def call(self, inputs, training=True):
        """
        net(x) net(x,  training=True): train mode
        net(x, training=Flase) : test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        x = inputs
        # gob
        print("input.shape:", x.shape)
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        print("state0:",state0)
        for word in tf.unstack(x, axis=1): #word:[b,100]  以中间的1维度展开（按照需要时序处理的维度展开，这里是句子长度）
            # print("words time", word)
            # x * wxh + h * whh 两层的
            out0, state0 = self.rnn_call0(word, state0, training)
            out1, state1 = self.rnn_call1(word, state1, training)
        # out:[b,64] => [b,1]
        x = self.outlayer(out1)
        prop = tf.sigmoid(x)

        return prop
def mainLowApi():
    unit = 64
    epochs = 4

    model = MyRNNLowAPI(unit)
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(0.001),
                  metrics=["acc"])
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.summary()
    model.evaluate(db_test)
    """
    Model: "my_rnn_low_api"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        multiple                  1000000   
    _________________________________________________________________
    simple_rnn_cell (SimpleRNNCe multiple                  10560     
    _________________________________________________________________
    simple_rnn_cell_1 (SimpleRNN multiple                  10560     
    _________________________________________________________________
    dense (Dense)                multiple                  65        
    =================================================================
    Total params: 1,021,185
    Trainable params: 1,021,185
    Non-trainable params: 0

    """


class MyRNN(keras.Model):
    """
    使用高级API 自定义的实现RNN
    """
    def __init__(self, units):
        super(MyRNN, self).__init__()

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len,
                                                   input_length=max_review_len)
        # [b, 80, 100] ,h_dim :64
        rnnlayer1 = tf.keras.layers.SimpleRNNC(units, dropout=0.5, return_sequences=True, unroll=True)
        rnnlayer2 = tf.keras.layers.SimpleRNN(units, dropout=0.5, unroll=True)

        self.rnn = keras.Sequential([
            rnnlayer1,
            rnnlayer2
            # tf.keras.layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            # tf.keras.layers.SimpleRNN(units, dropout=0.5, unroll=True)
        ])

        # fc, [b, 80, 100] => [b,1]
        self.outlayer = tf.keras.layers.Dense(1)
    def call(self, inputs, training=None, mask=None):
        #[b, 80]
        x = inputs
        # embedding [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        print("embedding",x.shape)
        # run cell compute x:[b, 80, 100] => [b, 64]
        x = self.rnn(x)
        print("rnn", x.shape)
        # out:[b, 64] =>[b,1]
        s = self.outlayer(x)
        print("outlayer",s.shape)
        # p(y is pos|x)
        prop = tf.sigmoid(s)
        print("sigmoid", prop.shape)
        return prop
def main():
    units = 64
    epochs = 4
    model = MyRNN(units)

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(db_train, epochs = epochs, validation_data=db_test)
    model.summary()
    """
    
    """
    model.evaluate(db_test)

def main_show_RNN_param():
    """
    使用高级API 实现RNN网络 观察每层的参数个数
    :return:
    """
    """
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 80, 100)           1000000   字典为10000个单词，词向量维度为100，故参数为100W
    _________________________________________________________________
    simple_rnn (SimpleRNN)       (None, 80, 64)            10560     一句80个词；RNN输入为100维（词向量为100），隐藏层为64（是不是可以理解为输出为64）；参数个数=（100+1）*64 + 64*64 = 10560
    _________________________________________________________________
    simple_rnn_1 (SimpleRNN)     (None, 64)                8256      RNN输入64维，输出为64维；参数个数= （64+1）*64 + 64*64 = 8256
    _________________________________________________________________
    dense (Dense)                (None, 1)                 65        输入64维，输出1维；
    =================================================================
    Total params: 1,018,881
    Trainable params: 1,018,881
    Non-trainable params: 0
    """
    embedding_dim = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, embedding_len,
                                   input_length=max_review_len),  # 字典为10000;词嵌入为100;句子长度80；
        # 第一层循环计算层：记忆体64个，每个时间步推送ht给下一层
        tf.keras.layers.SimpleRNN(embedding_dim, return_sequences=True, unroll=True),  #
        # 最后一个循环计算层 return_sequences用False, 其之前的每一个都设置为T
        tf.keras.layers.SimpleRNN(embedding_dim, unroll=True),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.summary()
    num_epochs = 1  # 10
    model.compile(loss=tf.losses.binary_crossentropy, optimizer="adam",
                  metrics=["acc"])
    history = model.fit(db_train,
                        epochs=num_epochs,
                        validation_data=db_test)

def main_show_RNN_each_layer_param_output():
    """
    使用高级API 实现RNN网络 观察每层的参数个数
    :return:
    """
    """
    
    """
    embedding_dim = 64
    embedding_layer = tf.keras.layers.Embedding(total_words, embedding_len,
                                   input_length=max_review_len)
    RNN_layer_1 = tf.keras.layers.SimpleRNN(embedding_dim, return_sequences=True, unroll=True)
    RNN_layer_2 = tf.keras.layers.SimpleRNN(embedding_dim, unroll=True)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential([
        embedding_layer,
        RNN_layer_1,
        RNN_layer_2,
        output_layer
    ])
    model.summary()
    num_epochs = 1  # 10
    model.compile(loss=tf.losses.binary_crossentropy, optimizer="adam",
                  metrics=["acc"])
    history = model.fit(db_train,
                        epochs=num_epochs,
                        validation_data=db_test)

    print(model.weights)

    print("*"*20)

    print(RNN_layer_1.output)
    print(RNN_layer_1.states)

def main_show_LSTM_param():
    """
    使用高级API 实现RNN网络 观察每层的参数个数
    :return:
    """

    embedding_dim = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, embedding_len,
                                   input_length=max_review_len),  # 字典为10000;词嵌入为100;句子长度80；
        # 第一层循环计算层：记忆体64个，每个时间步推送ht给下一层
        tf.keras.layers.LSTM(embedding_dim, return_sequences=True, unroll=True),  #
        # 最后一个循环计算层 return_sequences用False, 其之前的每一个都设置为T
        tf.keras.layers.LSTM(embedding_dim, unroll=True),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.summary()
    num_epochs = 1  # 10
    model.compile(loss=tf.losses.binary_crossentropy, optimizer="adam",
                  metrics=["acc"])
    history = model.fit(db_train,
                        epochs=num_epochs,
                        validation_data=db_test)

def main_show_GRU_param():
    """
    使用高级API 实现RNN网络 观察每层的参数个数
    :return:
    """
    embedding_dim = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, embedding_len,
                                   input_length=max_review_len),  # 字典为10000;词嵌入为100;句子长度80；
        # 第一层循环计算层：记忆体64个，每个时间步推送ht给下一层
        tf.keras.layers.GRU(embedding_dim, return_sequences=True, unroll=True),  #
        # 最后一个循环计算层 return_sequences用False, 其之前的每一个都设置为T
        tf.keras.layers.GRU(embedding_dim, unroll=True),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.summary()
    num_epochs = 1  # 10
    model.compile(loss=tf.losses.binary_crossentropy, optimizer="adam",
                  metrics=["acc"])
    history = model.fit(db_train,
                        epochs=num_epochs,
                        validation_data=db_test)


if __name__ == "__main__":
    # main()
    # mainLowApi()
    main_show_RNN_each_layer_param_output()