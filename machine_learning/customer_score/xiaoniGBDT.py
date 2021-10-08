# -*- coding: utf-8 -*-
"""
@Time     :2021/7/20 21:23
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

import random
import pandas as pd
import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier,RandomForestRegressor,RandomForestClassifier

if __name__ == "__main__":

    train_pos, train_neg, test_pos, test_neg = 0, 0, 0, 0
    train_x, train_y, test_x, test_y = [], [], [], []
    train_pos_x, train_pos_y, test_pos_x, test_pos_y = [], [], [], []
    train_neg_x, train_neg_y, test_neg_x, test_neg_y = [], [], [], []
    for line in open("./data/feature.txt", "r", encoding="utf-8"):
        line = line.strip()
        tokens = line.split("\t")
        # print(len(tokens))
        # print(tokens)
        if len(tokens) != 280:
            print (len(tokens))

        fea_array = [float(t) for t in tokens[2:]]
        label = int(tokens[1])


        ## 拆分测试集训练集 可以直接使用tarin_test_split()
        if random.random() < 0.8:
            train_x.append(fea_array)
            train_y.append(label)
            if label > 0.0001:
                train_pos += 1
                train_pos_x.append(fea_array)
                train_pos_y.append(label)
            else:
                train_neg += 1
                train_neg_x.append(fea_array)
                train_neg_y.append(label)
                # neg_dis_map[fea_array[123]] = neg_dis_map.get(fea_array[123],0) + 1
        else:
            test_x.append(fea_array)
            test_y.append(label)
            if label > 0.0001:
                test_pos += 1
                test_pos_x.append(fea_array)
                test_pos_y.append(label)
            else:
                test_neg += 1
                test_neg_x.append(fea_array)
                test_neg_y.append(label)
    print("训练集样本个数={} \n测试集的样本个数={}".format(len(train_x),len(test_x)))
    print("训练集正样本、负样本；测试集正样本、负样本个数：")
    print(train_pos, train_neg, test_pos, test_neg)

    # 手动shuffle 训练集
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(train_x)))
    train_x = train_x[shuffle_indices]
    train_y = train_y[shuffle_indices]

    cls = GradientBoostingRegressor()
    clf = GradientBoostingRegressor(max_depth=7, random_state=0, n_estimators=50)
    clf.fit(train_x, train_y)
    joblib.dump(clf, "random_forest_reg.m")
    Estimators = clf.estimators_
    # for index, model in enumerate(Estimators):
    #     filename = './img/iris_' + str(index) + '.pdf'
    # 保存数
    # Estimators = clf.estimators_
    # for index, model in enumerate(Estimators):
    #    filename = './img/iris_' + str(index) + '.pdf'
    #    dot_data = tree.export_graphviz(model , out_file=None,
    #                     feature_names=fea_list,
    #                     class_names=[0,1],
    #                     filled=True, rounded=True,
    #                     special_characters=True)
    #    graph = pydotplus.graph_from_dot_data(dot_data)
    # 使用ipython的终端jupyter notebook显示。
    #    Image(graph.create_png())
    #    graph.write_pdf(filename)


    clf = joblib.load("random_forest_reg.m")
    # print("Train score:%s" % (clf.score(train_x, train_y)))
    # print("test score:%s" % (clf.score(test_x, test_y)))
    weight = clf.feature_importances_
    # print(weight)
    for i, value in enumerate(weight.tolist()):
        print(i,value)