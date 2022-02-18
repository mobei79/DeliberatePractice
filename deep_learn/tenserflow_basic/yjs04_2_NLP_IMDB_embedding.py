# -*- coding: utf-8 -*-
"""
@Time     :2022/2/16 13:31
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

"""
词嵌入
    经过词条化、序列化，得到代表句子的数字序列矩阵；这就是数字化的语料库。
    如何对这些数字进行情感分析?
        我们需要从语料库中学习关键的信息，类似卷积神经网络提取图片中的关键图片特征；
        嵌入的核心是将所有相关的词汇，聚类为多维空间中的向量；
            以IMDB评论情感分类为例：tensorflow建立一个嵌入，来讲代表不同类型评论的词语进行聚类。
            数据集包含50000条电影评论，包含正面和负面评论；
"""
"""
加载数据集 各包含25000数据
    tf.enable_eager_execution tf1.0版本需要执行这个代码
    pip install -q tensorflow-datasets    安装数据集
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
# 加载数据
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb["test"]
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
for s,l in train_data:  # s和l都是张量，使用numpy()方法获取值
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())
for s,l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())
# 张量表示如下
# tf.Tensor(b"imdb love do", shape=(), dtype='string')
# tf.Tensor(1,dtype="int64")
# 训练时希望 标签 是numpy数组的形式，需要转化
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size= 10000# 所有超参数，便于修改优化
oov_tok="<OOV>"
max_length=120
trunc_type="post"
embedding_dim = 16

# 词条化 序列化
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)  # 给出词典大小和未登录词表示方法
tokenizer.fit_on_texts(training_sentences)                      # 对训练数据 创建词典
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)    # 根据词典编码对句子进行序列化
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type) # 原始句子长度不一致，需要阶段和补齐到固定长度

# 处理测试数据
test_sequences = tokenizer.texts_to_sequences(testing_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

"""
构建模型
    嵌入层是进行情感分析的关键；
    嵌入层原理概述：
        句子中意思相近的单词，往往彼此之间距离比较近；电影评论中枯燥、乏味多一起出现；
        因此我们可以在高维空间找到一组相似的向量，来表示情感相同的单词（注意是每个单词有一个向量），这些向量因为相似的标签而聚集在一起，
        所以神经网络可以学习这些向量，建立向量和标签之间的联系。向量成为了单词和单词情感之间的联系纽带。
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),# 嵌入层的结果是一个二维的数组，句子长度*向量维度
    keras.layers.Flatten(), # 平坦层展平
    keras.layers.GlobalAveragePooling1D(), # 全局平均池化层， 在每个向量的维度上取平均值输出，得到模型更加简洁，速度更快
    keras.layers.Dense(6, activation="relu"),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
# compile 编译模型
model.compile(loss=tf.losses.binary_crossentropy, optimizer='adam', metrics=["acc"]) # optimizer常用adam表示
model.summary() # 模型摘要信息

num_epochs = 1
model.fit(padded,
          training_labels_final,
          epochs=num_epochs,
          validation_data=(test_padded, testing_labels_final))

# 深入嵌入  将词典中10000个单词
e = model.layers[0] # 得到神经网络0层权值
weights = e.get_weights()[0]
print(weights.shape)    # shape(vocab_size, embedding_dim) (10000, 16)

"""
可视化 需要进入http://projector.tensorflow.org/ load data
    通过这个例子可以直观的感受 如何将单词映射到矢量空间。
"""
# 改变键值对中的顺序。为了方便可视化，将更改后的word_index以及嵌入层权重分别写入
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()] )
out_v = io.open("vecs.tsv", "w", encoding="utf-8")
out_m = io.open("meta.tsv", "w", encoding="utf-8")
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write("\t".join([str(x) for x in embeddings]) + '\n') # 单词的向量值
out_v.close()
out_m.close()
