# -*- coding: utf-8 -*-
"""
@Time     :2021/7/19 0:19
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import sklearn
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer #乳腺癌
from sklearn.datasets import load_diabetes #糖尿病
from sklearn.datasets import load_wine
from sklearn.datasets import load_boston
from sklearn.datasets.samples_generator import make_classification

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve # 可视化学习的整个过程，怎样降低
from sklearn.model_selection import validation_curve # 验证曲线
from sklearn.model_selection import train_test_split # 样本数据集拆分
from sklearn.model_selection import cross_validate #
from sklearn.model_selection import cross_val_predict # 交叉验证 - 返回是一个使用交叉验证以后的输出值
from sklearn.model_selection import cross_val_score # 交叉检验  -  1.9之后取消cross_validation改为model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn.metrics import r2_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier # K近邻
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.tree import DecisionTreeRegressor # 回归树
from sklearn.tree import export_graphviz # 	将生成的决策树导出为DOT格式，画图专用
from sklearn.tree import ExtraTreeClassifier # 高随机版本的分类树
from sklearn.tree import ExtraTreeRegressor # 高随机版本的回归树
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

import matplotlib
import matplotlib.pyplot as plt
import graphviz
import numpy as np
import pandas as pd
from scipy import stats
# import seaborn as sns

# gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=5, subsample=1
#                                  , min_samples_split=2, min_samples_leaf=1, max_depth=3
#                                  , init=None, random_state=None, max_features=None
#                                  , alpha=0.9, verbose=0, max_leaf_nodes=None
#                                  , warm_start=False
#                                  )
# train_feat = np.array([[1, 5, 20],
#                        [2, 7, 30],
#                        [3, 21, 70],
#                        [4, 30, 60],
#                        ])
# train_id = np.array([[1.1], [1.3], [1.7], [1.8]]).ravel()
# test_feat = np.array([[5, 25, 65]])
# test_id = np.array([[1.6]])
# print(train_feat.shape, train_id.shape, test_feat.shape, test_id.shape)
# gbdt.fit(train_feat, train_id)
# pred = gbdt.predict(test_feat)
# total_err = 0
# for i in range(pred.shape[0]):
#     print(pred[i], test_id[i])
#     err = (pred[i] - test_id[i]) / test_id[i]
#     total_err += err * err
# print(total_err / pred.shape[0])

from sklearn.datasets import load_diabetes #糖尿病

diabetes = load_diabetes()

X, y = load_diabetes().data, load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=5)



print(X.shape,y.shape) # (442, 10)
train = pd.DataFrame(X_train)


print("first 15 clo:", list(train.columns[:20]))
print(diabetes.keys())
# print("last 15 clo:".format(list(X.columns[-20:])))
print("*"*15+"样本数据的统计指标") #看数据是否被处理
print(train.describe())

print("*"*15+"数据缺失值")
print(pd.isnull(train).values.any())

print("*"*15+"样本数据的信息")
print(train.info())

cat_features =list(train.select_dtypes(include=['object']).columns)
cont_features =[cont for cont in list(train.select_dtypes(include=['float64','int64']).columns) if cont not in ["loss",'id']]

print("*"*15+"属性个数")
cat_uniques = []
for cat in cat_features:
    cat_uniques.append(len(train[cat].unique()))

