# -*- coding: utf-8 -*-
"""
@Time     :2021/11/26 16:41
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import numpy as np
filepath = "./sigmoid.csv"
xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
print(type(xy))
# print(xy[:,:-1])
# print(xy[:, [-1]])