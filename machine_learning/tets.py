# -*- coding: utf-8 -*-
"""
@Time     :2021/7/20 12:25
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import pandas as pd
with open("../../../../../Program Files/pythonwork/DeliberatePractice/machine_learning/data/base_feature.txt",
          "r") as f:
    data = f.readlines()
train = []
for i in range(len(data)):

    data[5].split("\t")
    train.append(data[5].split("\t"))

a = pd.DataFrame(train)
print(a.shape)