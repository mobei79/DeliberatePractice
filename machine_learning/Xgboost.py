# -*- coding: utf-8 -*-
"""
@Time     :2020/12/18 15:50
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import os
import sys
import pandas as pd
import numpy as np

import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# 原生的xgboost接口
from sklearn.datasets.samples_generator import make_classification
# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，
#输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                           n_clusters_per_class=1, n_classes=2, flip_y=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

params = {'max_depth':5, 'eta':0.5, 'verbosity':1, 'objective':'binary:logistic'}
raw_model = xgb.train(params, dtrain, num_boost_round=20)
pred_train_raw = raw_model.predict(dtrain)

for i in range(len(pred_train_raw)):
    if pred_train_raw[i]>0.5:
        pred_train_raw[i] = 1
    else:
        pred_train_raw[i] = 0
print(accuracy_score(dtrain.get_label(),pred_train_raw))
# >>> 0.9577333333333333

# TODO[使用sklearn风格接口，使用原生参数
sklearn_model_raw = xgb.XGBClassifier(**params)
sklearn_model_raw.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",
        eval_set=[(X_test, y_test)])

# TODO[使用sklearn风格接口，使用sklearn风格参数



# raw_data = pd.read_csv('train_modified.csv')
# print(raw_data.columns)
# print(raw_data.shape)