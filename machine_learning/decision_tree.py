# -*- coding: utf-8 -*-
"""
@Time     :2020/12/17 18:47
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

# 导入类库
from sklearn.datasets import load_iris
from sklearn import tree
import sys
import os
os.environ['path'] += os.pathsep+'D:\software\graphviz\Graphviz 2.44.1\\bin'

def test():
    # 导入sciki—learn自带的数据，有决策树你和 得到模型
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)

    # 将模型存入dot文件
    with open("iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    # 模型可视化 方法1
    import pydotplus
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("iris-1.pdf")

    # 模型可视化 方法2
    from IPython.display import Image
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())


# 一个实例
def decision_tree_example():
    from itertools import product

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as  plt

    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier

    # 使用自带的数据
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target

    # 训练模型 限制最大深度4
    clf = DecisionTreeClassifier(max_depth=4)
    # 拟合模型
    clf.fit(X, y)

    # 画图
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()


def decision_tree_vi(iris, clf):
    from IPython.display import Image
    from sklearn import tree
    import pydotplus
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())

if __name__ == '__main__':
    decision_tree_example()