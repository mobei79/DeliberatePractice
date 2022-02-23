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
摘要：
    1. 在yjs=0.4部分，处理NLP问题时。是使用tensorflow的data services 来加载IMDB评论数据集，并设计了一个分类器；
    但是这种标准的数据集，也有已经经过预处理的数据集版本。比如："imdb_reviews/subwords8k"，IMDB数据集已经以子词的形式进行了预先词条化，

    2. 子词对分类器的影响？
        如果直接使用之前模型（Embedding+Flatten+Dense）进行训练，结论为效果并不好。
        原因：我们使用的是子词，本身没有实际意义，只有按照一定顺序排列才有意义，所以需要序列的方式进行学习。这也引出了之后的循环神经网络
    3. 循环神经网络
        子词出现的顺序，对于理解单词的含义非常重要。
        例子：斐波那切数列
        循环神经网络在处理文本分类时，存在一个问题？
        参考一个词语的上下文语境，对于理解词语的含义有帮助。一种先进的循环神经网络LSTM被提出用来分析文本上下文含义。
            增加cell state结构，实现长期记忆，可以对于上下文语义有很大帮助。
            cell state也可以是双向的，如后面的内容也影响前面的状态。
    4. 含卷积层的RNN，含门控循环单元GRU.
    
"""
import tensorflow as tf
import numpy as np
import io

# from tensorflow import keras
print(tf.__version__)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import json
import tensorflow_datasets as tfds

# 加载数据 这里使用的是subword8k版本，也就是子词数据集
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb["test"]
tokenizer = info.features['text'].encoder  # 通过这个语句获得一个子词的分词器，这是一个预训练好的子词分类器。【这里的数据是已经分词后的】
# print(tokenizer.subwords)  # 查看分词器的词汇表
print(tokenizer.vocab_size)
# 查看她如何对字符串进行编码的
# sample_string = "TensorFlow, from basics to mastery"
# tokenized_string = tokenizer.encode(sample_string)  # 编码
# print(tokenized_string)
# original_string = tokenizer.decode(tokenized_string)  # 解码
# print(original_string)
# for ts in tokenized_string:  # 查看编码具体内容
#     print("{}------>{}".format(ts, tokenizer.decode([ts])))


BUFFER_SIZE = 1000 #25000
BATCH_SIZE = 1
train_data = train_data.shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)
"""
构建神经网络模型
"""
# embedding_dim = 64
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),  # 嵌入层的结果是一个二维的数组，句子长度*向量维度
#     # keras.layers.Flatten(),  # 平坦层展平
#     keras.layers.GlobalAveragePooling1D(),  # 全局平均池化层， 在每个向量的维度上取平均值输出，得到模型更加简洁，速度更快;有些编码也难以展平
#     keras.layers.Dense(6, activation="relu"),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
# model.summary()


""" LSTM
    使用tf.keras.layers.LSTM(64)来实现LSTM层，64为LSTM输出维度；
    Bidirctional 使得LSTM可以记忆两个方向上下文信息，双向LSTM层输出维度为128;
    当LSTM层衔接时，需要设置前一个LSTM的return_sequences=True，可以确保上一层输出，可以与下一层输入相匹配。

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 64)          523840    
_________________________________________________________________
bidirectional (Bidirectional (None, None, 128)         66048     
_________________________________________________________________
bidirectional_1 (Bidirection (None, 64)                41216     
_________________________________________________________________
dense (Dense)                (None, 64)                4160      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65        
=================================================================
Total params: 635,329
Trainable params: 635,329
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code -1

"""
# embedding_dim = 64
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim), #vocab_size=8185, 输出8185*64
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)), #
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     # tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(64, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
# model.summary()


"""RNN
    Embedding函数的三个参数： 本质上是对输入数据降维的过程；
        input_dim: 指输出输入数据的维度【字典的维度】，也就是一个单词是
        output_dim:
        input_length:

"""
# embedding_dim = 64
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(tokenizer.vocab_size, 100, input_length=80),  # vocab_size=8185, 输出8185*64
#     tf.keras.layers.SimpleRNN(64, return_sequences=True, unroll=True),  #
#     tf.keras.layers.SimpleRNN(64, unroll=True),
#     # tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(64, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
# model.summary()

"""
模型训练
"""
num_epochs = 1 #10
model.compile(loss=tf.losses.binary_crossentropy, optimizer="adam",
              metrics=["acc"])
history = model.fit(train_data,
                    epochs=num_epochs,
                    validation_data=test_data)


"""
做图显示结果
    1层LSTM、2层LSTM、
        10次训练，50次训练
            acc loss
            
"""
import matplotlib.pyplot as plt
def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_"+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
plot_graph(history, "acc")
plot_graph(history, "loss")

"""
1. 先比较了拥有一层LSTM和两层LSTM的模型精度，所有模型都训练10个周期。
    我们可以看到训练准确度的曲线是比较类似的，但是双层LSTM的训练准确度曲线更加平滑，同时双层LSTM的验证曲线更好；
    在观察损失函数曲线：结论基本一致。
2. 训练50个周期以后
    我们看到单层的网络，训练准确度虽然总体在上升，但是在某些地方出现了急剧的下降，这说明算法的鲁棒性不高；
    而双层模型准确度曲线非常平滑，说明训练过程更加稳定；
3. 再看一下验证数据表现；
    先看准确性：两者基本都在80%附近，双层网络更好一些，由于我们的验证集中，有很多未登录词，所以80的验证准确率基本令人满意，
    损失曲线：同样双层网络的训练损失曲线更加平滑，但是验证损失曲线，随着训练周期的增加，不断上升，
        后期可以增加训练周期数量，看验证损失曲线能否趋于平稳。
"""

