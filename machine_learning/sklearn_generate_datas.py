# -*- coding: utf-8 -*-
"""
@Time     :2021/7/21 11:39
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
# 糖尿病数据集，回归使用。样本数据集的特征默认是一个(442, 10)大小的矩阵，样本值是一个包含442个数值的向量。
from sklearn.datasets import load_breast_cancer
#手写体数据，分类使用。每个手写体数据使用8*8的矩阵存放。样本数据为(1797, 64)大小的数据集。
from sklearn.datasets import load_digits
# 波士顿房价数据，回归使用。样本数据集的特征默认是一个(506, 13)大小的矩阵，样本值是一个包含506个数值的向量。
from sklearn.datasets import load_boston


# https://blog.csdn.net/wangdong2017/article/details/81326341
'''
sklearn.datasets.make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,  
                    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,  
                    flip_y=0.01, class_sep=1.0, hypercube=True,shift=0.0, scale=1.0,   
                    shuffle=True, random_state=None) 
通常用于分类算法。 
n_features :特征个数= n_informative信息 + n_redundant冗余 + n_repeated重复 
n_informative：多信息特征的个数 
n_redundant：冗余信息，informative特征的随机线性组合 
n_repeated ：重复信息，随机提取n_informative和n_redundant 特征 
n_classes：分类类别 
n_clusters_per_class ：某一个类别是由几个cluster构成的
'''
from sklearn.datasets import make_classification
# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，
#输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X.shape, y.shape)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# datas = load_breast_cancer()
# print(datas.data.shape)
# print(datas.target)




