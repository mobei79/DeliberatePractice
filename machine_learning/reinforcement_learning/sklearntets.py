# -*- coding: utf-8 -*-
"""
@Time     :2021/7/12 14:00
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve # 可视化学习的整个过程，怎样降低
from sklearn.model_selection import train_test_split # 样本数据集拆分
from sklearn.model_selection import cross_validate #
from sklearn.model_selection import cross_val_predict # 交叉验证 - 返回是一个使用交叉验证以后的输出值
from sklearn.model_selection import cross_val_score # 交叉检验  -  1.9之后取消cross_validation改为model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier # K近邻
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def kneighbors():
    iris = load_iris()
    X = iris.data
    Y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=4)
    knn = KNeighborsClassifier(n_neighbors=5)

    ### 单次训练
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    # print(knn.score(X_test, y_test))

    ### K折交叉检验 - 分数
    # score = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
    # print(score.mean())

    ### K折交叉检验 - 字典结构
    ### dict_keys([‘fit_time’, ‘score_time’, ‘test_score’, ‘train_score’])，
    ### 表示的是模型的训练时间，测试时间，测试评分和训练评分。
    ### 用两个时间参数和两个准确率参数来评估模型，这在我们进行简单的模型性能比较的时候已经够用了。
    # cv_result = cross_validate(knn, X, Y, cv=10)
    # print(cv_result['test_score'].mean())
    # print(cv_result)
    ### K折交叉检验 - 预测值
    # rst = cross_val_predict(knn, X, Y, cv=10)
    # print(rst)

    ## 使用不同参数训练，观察参数性能，选择最优参数
    k_range = range(1,31)
    k_score = []
    for k in k_range:
        knn_s = KNeighborsClassifier(n_neighbors=k) # K越小，异常值影响大，会过拟合； K越大，异常值影响越小，会欠拟合；
        scores = cross_val_score(knn_s, X, Y, cv=10, scoring='accuracy') # Classification 使用accuracy
        # loss = -cross_val_score(knn_s, X, Y, scoring='mean_squared_error') # 线性回归regression 判断误差
        k_score.append(scores.mean())
    plt.plot(k_range, k_score)
    plt.xlabel("Value of k for KNN")
    plt.ylabel("Cross-validated Accuracy")
    plt.show()

# print(sklearn.metrics.SCORERS.keys()) # 输出所有的评分值
def svcl():
    digits = load_digits()
    X = digits.data
    y = digits.target
    train_sizes, train_loss, test_loss = learning_curve(
        SVC(gamma=0.001),X,y,cv=10,scoring="neg_mean_squared_error",
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1]) #记录的点时学习过程的10% 25% 。。。记录值
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    plt.plot(train_sizes, train_loss_mean, "o-", color="r",label="Training")
    plt.plot(train_sizes, test_loss_mean, "o-", color="g",label="Cross-validation")
    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()


def logistss():
    iris = load_iris()
    X = iris.data[:,:2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=0)
    # 设置随机数种子，以便比较结果

    print(y_test)
    print(X_test)

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    logreg = LogisticRegression(C=1e5)
    logreg.fit(X_train, y_train)

    prepro = logreg.predict(X_test_std)
    acc = logreg.score(X_test_std, y_test)

    print(prepro)
    print("*"*20)
    print(acc)


if __name__ == "__main__":
    # kneighbors()

    # svcl()
    logistss()