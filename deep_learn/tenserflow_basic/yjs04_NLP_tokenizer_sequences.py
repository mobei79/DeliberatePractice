# -*- coding: utf-8 -*-
"""
@Time     :2022/2/16 11:41
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
"""
摘要
    神经网络只能进行数字的复杂运算，不能直接处理文本数据，所以需要对文本数据进行转换；
    sentences（句子） - tokenizer（分词器-词条化） - sequences(序列化) - padding（句子补齐）
"""
import tensorflow as tf
import numpy as np
# from tensorflow import keras
print(tf.__version__)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
"""
分词器 - 词条化 = 构造器+编码
    keras提供了tokenizer，是一个分词器，可以快速产生“词典”，将文本转换为tokens流，将单词转换为数字，用于后续训练；
        标点对单词没有影响；带小写不敏感；

"""
sentences = [
    "I love my dog .",
    "I love my cat ,",
    "you love my dog!",
    "Do you think my dog is amazing?"
]
# 实例化Tokenizer
#   建立一个100个单词的“词典”；“分词器”选择“词频”最高的100个单词作为词典，并编码。
#   oov_token 用于代替”未登录词“
#   词频低的单词对神经网络训练影响较小，但是会极大地增加训练时间，需要仔细设置该参数。
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences) # 编码
word_index = tokenizer.word_index   # 输出键值对。
print("#"*10+"word_index ")
print(word_index)


"""
序列化
    将分词器 词条化的编码，基于编码将句子序列化，进一步对序列化的列表进行处理，使各个句子编码长度相同，才能用于神经网络训练
    
    在神经网络的训练和推断阶段 会使用相同的词典；语料越多，词典中的单词也就越多；
    不在字典中的单词会丢失，可以设置未登录词的唯一代替值OOV_token，来保证句子的结构；
"""
sequences = tokenizer.texts_to_sequences(sentences)
print("#"*10+"sequences ")
print(sequences)    #编码为整数的sentences数组
test_sentences = [
    "i really love my dog",
    "my dog loves my manatee"
]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
print("-"*10+"test_sentences "+"-"*10)
print(test_sequences)

"""
序列化后句子补齐
    因为神经网络的输入需要维度一致。
    分词器创建了序列sequences，传入pad_sequences进行补齐；padding="pre or post"控制在前面还是后面补齐
    maxlen控制句子长度；trucating="pre or post"控制在前面还是后面进行截断；
    
"""
padded = pad_sequences(sequences, padding='pre', truncating='post',maxlen=5)
print("-"*10+"padded "+"-"*10)
print(padded)


