# -*- coding: utf-8 -*-
"""
@Time     :2022/2/18 12:46
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
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
train_df["text_len"] = train_df["text"].apply(lambda x:len(x.split()))
print(train_df["text_len"].describe())