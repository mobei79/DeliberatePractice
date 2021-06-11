# -*- coding: utf-8 -*-
"""
@Time     :2020/12/21 16:37
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV    #网格搜索
from sklearn.model_selection import StratifiedKFold #交叉验证
from sklearn.model_selection import train_test_split #将数据集分开成训练集和测试集
from sklearn.model_selection import cross_validate
import matplotlib.pylab as plt

# 数据准备
train = pd.read_csv('./train_data/train_modified.csv')
target = 'Disbursed'    # 二元分类的输出
IDcol = 'ID'
print(train['Disbursed'].value_counts())

# 选择好样本特征和类别输出
x_columns = [x for x in train.columns if x not in [target,IDcol]]
X = train[x_columns]
y = train['Disbursed']

# 先使用默认值，拟合一次数据
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X, y)

# 查看使用默认参数拟合的效果，随机森林可以查看袋外错误率oob；其他的模型可以使用roc_auc_score之类的值
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X)#[:,1]  # 概率校准 置信度级别  http://codingdict.com/article/28426  https://www.cnblogs.com/pinard/p/6160412.html?utm_source=itdadao&utm_medium=referral
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob[:,1]))
"""
 输出如下，可见袋外分数已经很高，而且AUC分数也很高。RF的默认参数拟合效果对本例要好一些。 
 oob: 0.98315
 AUC Score (Train): 0.999994
 
 可以用来评价参数拟合效果的么？
"""

# 网格搜索 找到最优参数
'''
# 对n_estimators进行网格搜索
param_test1 = {'n_estimators':range(10,70,10)}
gsearch1 = GridSearchCV(estimator= RandomForestClassifier(min_samples_split=100,
                                                          min_samples_leaf=20,
                                                          max_depth=8,
                                                          max_features='sqrt',
                                                          random_state=10),
                        param_grid=param_test1,
                        scoring='roc_auc',
                        cv=5)
gsearch1.fit(X, y)  #进行网格搜索
print('Best:{0} using {1}'.format(gsearch1.best_score_,gsearch1.best_params_))
print(gsearch1.cv_results_)
means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))
"""
结果如下所示：
Best:0.8211334476626015 using {'n_estimators': 60}
*********
0.806809  with:   {'n_estimators': 10}
0.816003  with:   {'n_estimators': 20}
0.818183  with:   {'n_estimators': 30}
0.818384  with:   {'n_estimators': 40}
0.820341  with:   {'n_estimators': 50}
0.821133  with:   {'n_estimators': 60}
这样我们得到了最佳的弱学习器迭代次数，60
"""

# 接着我们对决策树最大深度max_depth 和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,
                                                           min_samples_leaf=20,
                                                           max_features='sqrt' ,
                                                           oob_score=True,
                                                           random_state=10),
                        param_grid = param_test2,
                        scoring='roc_auc',
                        iid=False,
                        cv=5)
gsearch2.fit(X,y)
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
"""
输出结果如下：
0.793792  with:   {'max_depth': 3, 'min_samples_split': 50}
0.793386  with:   {'max_depth': 3, 'min_samples_split': 70}
0.793503  with:   {'max_depth': 3, 'min_samples_split': 90}
0.793666  with:   {'max_depth': 3, 'min_samples_split': 110}
0.793874  with:   {'max_depth': 3, 'min_samples_split': 130}
0.793728  with:   {'max_depth': 3, 'min_samples_split': 150}
0.793776  with:   {'max_depth': 3, 'min_samples_split': 170}
0.793495  with:   {'max_depth': 3, 'min_samples_split': 190}
0.809604  with:   {'max_depth': 5, 'min_samples_split': 50}
0.809199  with:   {'max_depth': 5, 'min_samples_split': 70}
0.808878  with:   {'max_depth': 5, 'min_samples_split': 90}
0.809230  with:   {'max_depth': 5, 'min_samples_split': 110}
0.808229  with:   {'max_depth': 5, 'min_samples_split': 130}
0.808010  with:   {'max_depth': 5, 'min_samples_split': 150}
0.807923  with:   {'max_depth': 5, 'min_samples_split': 170}
0.807712  with:   {'max_depth': 5, 'min_samples_split': 190}
0.816876  with:   {'max_depth': 7, 'min_samples_split': 50}
0.818725  with:   {'max_depth': 7, 'min_samples_split': 70}
0.815011  with:   {'max_depth': 7, 'min_samples_split': 90}
0.814755  with:   {'max_depth': 7, 'min_samples_split': 110}
0.815571  with:   {'max_depth': 7, 'min_samples_split': 130}
0.814587  with:   {'max_depth': 7, 'min_samples_split': 150}
0.816010  with:   {'max_depth': 7, 'min_samples_split': 170}
0.817038  with:   {'max_depth': 7, 'min_samples_split': 190}
0.820904  with:   {'max_depth': 9, 'min_samples_split': 50}
0.819080  with:   {'max_depth': 9, 'min_samples_split': 70}
0.820357  with:   {'max_depth': 9, 'min_samples_split': 90}
0.818893  with:   {'max_depth': 9, 'min_samples_split': 110}
0.819913  with:   {'max_depth': 9, 'min_samples_split': 130}
0.817876  with:   {'max_depth': 9, 'min_samples_split': 150}
0.818976  with:   {'max_depth': 9, 'min_samples_split': 170}
0.817459  with:   {'max_depth': 9, 'min_samples_split': 190}
0.823952  with:   {'max_depth': 11, 'min_samples_split': 50}
0.823803  with:   {'max_depth': 11, 'min_samples_split': 70}
0.819527  with:   {'max_depth': 11, 'min_samples_split': 90}
0.822536  with:   {'max_depth': 11, 'min_samples_split': 110}
0.819503  with:   {'max_depth': 11, 'min_samples_split': 130}
0.818869  with:   {'max_depth': 11, 'min_samples_split': 150}
0.819104  with:   {'max_depth': 11, 'min_samples_split': 170}
0.815640  with:   {'max_depth': 11, 'min_samples_split': 190}
0.822908  with:   {'max_depth': 13, 'min_samples_split': 50}
0.821766  with:   {'max_depth': 13, 'min_samples_split': 70}
0.824154  with:   {'max_depth': 13, 'min_samples_split': 90}
0.824202  with:   {'max_depth': 13, 'min_samples_split': 110}
0.822085  with:   {'max_depth': 13, 'min_samples_split': 130}
0.818525  with:   {'max_depth': 13, 'min_samples_split': 150}
0.819546  with:   {'max_depth': 13, 'min_samples_split': 170}
0.820923  with:   {'max_depth': 13, 'min_samples_split': 190}
这样我们得到了最佳的 决策树最大深度max_depth 和内部节点再划分所需最小样本数min_samples_split
0.824202  with:   {'max_depth': 13, 'min_samples_split': 110}
"""
'''

# 　使用上述参数，我们看看我们现在模型的袋外分数
rf1 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=110,
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
rf1.fit(X,y)
print(rf1.oob_score_)
"""
此时的输出为：
0.984
可见此时模型的袋外分数基本没有提高，主要原因是0.984已经是一个很高的袋外分数了，如果想进一步需要提高模型的泛化能力，我们需要更多的数据。
"""

