# -*- coding: utf-8 -*-
"""
@Time     :2022/2/16 12:48
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
"""
摘要
    http://projector.tensorflow.org/
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json
"""
# import tensorflow as tf
# import numpy as np
# # from tensorflow import keras
# print(tf.__version__)
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import keras
# import json
#
# with open("D:\Program Files\pythonwork\DeliberatePractice\deep_learn\\tenserflow_basic\datas\sarcasm.json", "r") as f:
#     datastore = json.load(f)
# sentences = []
# labels = []
# urls = []
# for item in datastore:
#     sentences.append(item["headline"])
#     labels.append(item["is_sarcastic"])
#     urls.append(item["article_link"])
#
# tokenizer = Tokenizer(oov_token="<OOV>") # 不指定num_words参数返回所有单词，
# tokenizer.fit_on_texts(sentences)
# # word_index = tokenizer.word_index
# sequences =tokenizer.texts_to_sequences(sentences) # 创建序列
# padded = pad_sequences(sequences, padding='post')   #进行pad_sequences
# print(padded[0])
# print(padded.shape)




"""
上面是词条化和序列化
下面是词条化 序列化 词嵌入 模型训练
"""
import tensorflow as tf
import numpy as np
# from tensorflow import keras
print(tf.__version__)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import json

"""
通过修改参数，不断优化模型效果
"""
# vocab_size = 10000 #词典大小
vocab_size = 100 #词典大小
embedding_dim = 16
# max_length = 32
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000
"""
数据集包含27000条数据，
    20000条训练，7000条训练
"""
with open("D:\Program Files\pythonwork\DeliberatePractice\deep_learn\\tenserflow_basic\datas\sarcasm.json", "r") as f:
    datastore = json.load(f)
sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item["headline"])
    labels.append(item["is_sarcastic"])
    # urls.append(item["article_link"])

# 拆分训练数据和验证数据
training_sentences = sentences[0:training_size]
training_labels = labels[0:training_size]
testing_sentences = sentences[training_size:]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>") # 不指定num_words参数返回所有单词，
tokenizer.fit_on_texts(training_sentences)
# word_index = tokenizer.word_index
training_sequences =tokenizer.texts_to_sequences(training_sentences) # 创建序列
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                padding=padding_type,truncating=trunc_type)   #进行pad_sequences
testing_sequences =tokenizer.texts_to_sequences(testing_sentences) # 创建序列
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                                padding=padding_type,truncating=trunc_type)   #进行pad_sequences
# print(testing_padded[0])
# print(testing_padded.shape)

"""
创建神经网络
"""
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"]) # 因为是一个二分类器，所以用二进制交叉熵作为损失函数
model.summary()
num_epochs = 30
history = model.fit(training_padded, training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

"""
做图显示结果
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