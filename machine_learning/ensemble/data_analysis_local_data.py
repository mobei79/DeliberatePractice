# -*- coding: utf-8 -*-
"""
@Time     :2021/7/21 12:03
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import xgboost as xgb

import pandas as pd
import numpy as mp
from scipy import stats
import seaborn as sns
from copy import deepcopy
import numpy as np
#
# #解决中文显示问题
# matplotlib.rcParams["font.sans-serif"] = [u"SimHei"]
# matplotlib.rcParams["axes.unicode_minus"] = False

#拦截异常
# import warnings
# from sklearn.linear_model.coordinate_descent import ConvergenceWarning
# warnings.filterwarnings(action='ignore',category=ConvergenceWarning)

#解决模型参数显示问题 保留小数而非科学记数法
# np.set_printoptions(precision = 4, suppress = True)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
# pd.set_option("max_colwidth", 20)
#
# with open("../data/base_feature.txt", 'r') as f:
#     complex = f.readlines()
#
# data_xq = []
# for line in complex:
#     data_xq.append(line.strip().split('\t'))
#
# train = pd.DataFrame(data_xq)
# print(train.shape)
#
# print(train[:12])


import numpy as np

Array1 = [[1, 2, 3], [4, 5, 6]]
Array2 = [[11, 25, 346], [734, 48, 49]]
Mat1 = np.array(Array1)
Mat2 = np.array(Array2)

correlation = np.corrcoef(Mat1, Mat2)
print("矩阵1=\n", Mat1)
print("矩阵2=\n", Mat2)
print("相关系数矩阵=\n", correlation)

fig = plt.figure()

sns.heatmap(correlation, annot=True)


