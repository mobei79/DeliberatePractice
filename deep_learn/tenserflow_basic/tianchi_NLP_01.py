# -*- coding: utf-8 -*-
"""
@Time     :2022/2/18 12:46
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import matplotlib.pyplot as plt
"""
data set 

train_set.csv   训练集
test_a.csv      测试集
test_a_sample_submit.csv    测试集2

"""
tianchi_data_path = "D:\Program Files\pythonwork\dataset\DeliberatePractice\deep_learn\\tianchi\\"


import pandas as pd
train_df = pd.read_csv(tianchi_data_path+"train_set.csv",
                       sep="\t",        # csv文件每列的分割符号 \t
                       nrows=100)       # 读取行数
print(train_df.head())

"""
数据分析
    非结构化的数据不需要太多数据分析工作，看一下数据规律即可。如文本长度、类型分布、字符分布等等
    
"""
##-- 统计每个句子的长度，平均长度等等值。
train_df["text_len"] = train_df["text"].apply(lambda x:len(x.split()))
print(train_df["text_len"].describe())
"""
统计每个句子的长度，平均长度等等值。
------
count     100.000000
mean      872.320000
std       923.138191
min        64.000000
25%       359.500000
50%       598.000000
75%      1058.000000
max      7125.000000
Name: text_len, dtype: float64
"""
##-- 绘制句子长度直方图， 大部分句子长度都在2000以内。
_ = plt.hist(train_df["text_len"], bins=200)
plt.xlabel("Text char count!")
plt.ylabel("Histogram of char count!")
plt.show()

##-- 对数据集的类别进行分析统计。具体的统计每个类别新闻的样本个数。
train_df["label"].value_counts().plot(kind="bar")
plt.xlabel("news class count")
plt.ylabel("category")
plt.show()
"""
print(train_df["label"].value_counts())
数据集的label标签对应关系如下：
    {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 
    7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}
    从统计结果来看，数据集分布不均匀，科技类新闻最多，股票类次之，星座类新闻最少
"""

##-- 字符分布统计 -
from collections import Counter
all_lines = ''.join(train_df['text'])
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse=True)
print(len(word_count))
print(word_count[0])
print(word_count[-1])
"""
每个字符出现的次数
2493
('3750', 3702)
('5034', 1)
"""

##-- 统计每个符号出现的次数
"""
可以根据字在每个句子中出现的情况，反推出标点符号
    字符3750 900 648的新闻覆盖率接近99很可能是标点符号
"""
train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' '))))) # set()创建无序不重复元素集合，可以进行关系测试，删除重复元素，计算交集并集等等
# 将每个句子切分split， 去重set，转为list，在join为一个string，存入新的列text_unique【这时候包含了每个字符串内唯一出现的字符，现在某个字符出现的次数就是这个字符在句子覆盖率】
#
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
# print(word_count.items())
print(sorted(word_count.items(), reverse=True))
print(sorted(word_count.items(), key=lambda d:d[0], reverse=True))
print(sorted(word_count.items(), key=lambda d:int(d[1]), reverse=True))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse=True)
# print(word_count[:10])
# print(len(word_count))
# print(word_count[0])
# print(word_count[1])
# print(word_count[1])
# print(word_count[-1])


"""
数据分析结论：
    1 每个新闻包含字符个数平均为1000个左右，有的新闻超过2000
    2 新闻类型分布不均匀，科技新闻样本约4万，星座类新闻不足1k；
    3 数据集总共包含7000-8000个字符
    从而：
    1 每个新闻字符个数较多，需要进行截断
    2 类别不均衡，会影响模型精度。


本章作业¶
假设字符3750，字符900和字符648是句子的标点符号，请分析赛题每篇新闻平均由多少个句子构成？
统计每类新闻中出现次数对多的字符
"""