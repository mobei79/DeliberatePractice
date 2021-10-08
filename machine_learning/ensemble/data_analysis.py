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
import numpy as np
from scipy import stats
import seaborn as sns
from copy import deepcopy
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("max_colwidth", 30)
np.set_printoptions(threshold=100)

# params = {
#
# }
# xgb_model = xgb.XGBClassifier(**params)

boston = load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)
print(X.shape, y.shape) #(506, 13) (506,)
# print(y)
"""
# 查看样本统计数据
主要看均值，标准差，最大细小值
"""
train = pd.DataFrame(X)
# print(train.describe())

# 查看缺失值
# print(pd.isnull(train).values.any())

# 连续值与离散值。数据特征观察
# print(train.info())

## 查看连续特征和离散特征
cat_features = list(train.select_dtypes(include=['object']))
# print('离散特征Categorical: {} features'.format(len(cat_features)))
cont_features = [cont for cont in list(train.select_dtypes(include=['float64','int64'])) if cont not in ['loss','id']]
# print('连续特征Continuous: {} features'.format(len(cont_features)))

#统计类别属性中不同类别的个数
cat_uniques=[]
for cat in cat_features:
    cat_uniques.append(len(train[cat].unique()))
# uniq_values_in_categories = pd.DataFrame.from_items([('cat_name',cat_features),('unique_values',cat_uniques)])
# uniq_values_in_categories.head()


# plt.subplots(figsize=(16,9))
# correlation_mat = train[cont_features].corr()
# sns.heatmap(correlation_mat,annot=True)

"""
特征工程
"""
# 特征选择 过滤法
# 样本标准化 Z-Score；(x-mean)/std
from sklearn import preprocessing
train_scaled = preprocessing.scale(train)
sc = preprocessing.StandardScaler()
train_scaled2 = sc.fit_transform(train)
train_scaled = pd.DataFrame(train_scaled)
train_scaled2 = pd.DataFrame(train_scaled2)
print(train[:12])
print(train_scaled[:12])
print(train_scaled2[:12])

## min - max
min_max_sc = preprocessing.MinMaxScaler()
min_max_scaler = min_max_sc.fit_transform(train)
print(min_max_scaler)

# 正则化
nomal = preprocessing.normalize(train,norm='l2')
print(nomal)