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
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes #糖尿病

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve # 可视化学习的整个过程，怎样降低
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

from sklearn.metrics import r2_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier # K近邻
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#解决中文显示问题
matplotlib.rcParams["font.sans-serif"] = [u"SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

#拦截异常
# import warnings
# from sklearn.linear_model.coordinate_descent import ConvergenceWarning
# warnings.filterwarnings(action='ignore',category=ConvergenceWarning)

#解决模型参数显示问题 保留小数而非科学记数法
np.set_printoptions(precision = 4, suppress = True)

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


def logistic_regression_breast_cancer():

    ### 数据导入 ：使用网页加载 或者sklearn.datasets
    # column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
    #                 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
    #                 'Normal Nucleoli', 'Mitoses', 'Class']
    # data = pd.read_csv(
    #     'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin//breast-cancer-wisconsin.data',
    #     names=column_names)
    # data = data.replace(to_replace='?', value=np.nan)  # 非法字符的替代
    # data = data.dropna(how='any')  # 去掉空值，any：出现空值行则删除
    # print(data.shape)
    # print(data.head())

    canver = load_breast_cancer()
    X = canver.data
    y = canver.target
    print("数据规模：{}".format(X.shape))

    ### 数据规则化 查分样本数据  ---》 有这里fit和transform的关系，可知，fit是用来拟合样本数据的，拟合得到均值和方差，之后就可以一直使用
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=5)
    # 设置随机数种子，以便比较结果
    sc = StandardScaler() # #标准化数据，保证每个维度的特征数据方差为1，均值为0.使得预测结果不会被某些维度过大的特征值主导
    # sc.fit(X_train)  #先拟合
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)
    x_train_std = sc.fit_transform(X_train) # #先拟合，在转换
    X_test_std = sc.transform(X_test) ##上面拟合过，这里直接转换

    ### 构建模型 并训练
    LR = LogisticRegression(C=1e5,penalty='l2',tol=0.1)  #调用逻辑回归模型，里面的参数可以自己设置，通过交叉验证来判断最优参数，我前面文章有介绍 使用l2penalization也可以
    print("模型参数：{}".format(LR.get_params()))
    LR.fit(x_train_std, y_train)
    lr_predict = LR.predict(X_test_std) # 预测值分类结果
    print("实际结果为：{}".format(y_test))
    print("预测结果为：{}".format(lr_predict))
    lr_probs = LR.predict_proba(X_test_std)  # 预测为各个值的概率; 输出2-D数组，结果预测为0或1的概率
    print("逻辑回归模型输出的概率值：".format(lr_probs))

    ### 模型评估1 - 准确率等
    acc = LR.score(X_test_std, y_test)      # 使用模型的评分函数
    acc_1 = accuracy_score(y_test, lr_predict) # 使用sklearn计算库中的准确率计算函数
    precision = precision_score(y_test, lr_predict)
    recall = recall_score(y_test, lr_predict)

    print("预测准确度为：{}".format(acc))
    print("预测准确度为：{}".format(acc_1))
    print("预测精准率为：{}".format(precision))
    print("预测召回率为：{}".format(recall))

    ## 混淆矩阵
    """
    混淆矩阵
    从0到1
    对角线表示分类正确的，求和除以样本总数得到模型的准确率accuracy【metrics.accuracy_score(y_test, lr_predict)】【LR.score(y_test, lr_predict)】
    """
    plot_confusion_matrix(LR, X_test_std, y_test)
    plt.show()

    """
    ROC曲线
    FPR做横轴 - 错判为正的数目占所有实际负的比例;
    TPR做纵轴 - 判断为正的数目占所有实际正的比例；
    TPR越高，FPR越小，我们的模型和算法就越高效；
    """
    print("*"*17 + "ROC曲线" + "*"*19)
    lr_probs = LR.predict_proba(X_test_std)  # 预测为各个值的概率; 输出2-D数组，结果预测为0或1的概率
    fpr, tpr, thresholds =  roc_curve(y_test, lr_probs[:, 1], pos_label=1) # pos_label被认为是积极的标签
    plt.title("ROC - curve")
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()
    ### AUC值
    # auc_score0 = auc(fpr, tpr)
    # auc_score1 = roc_auc_score(y_test, lr_probs[:, 1])
    # print("auc_score0 = {}".format(auc_score0))
    # print("auc_score1 = {}".format(auc_score1))
    # display = plot_roc_curve(LR, X_test_std, y_test) #
    # print('type(display):',type(display))
    # plt.show()


    ### 得到模型的PR曲线
    print("*" * 17 + "PR曲线" + "*" * 19)
    precision, recall, thresholds = precision_recall_curve(y_test, lr_probs[:, 1])  # 检索模型结果为1的概率值
    plt.plot(recall, precision)
    plt.title("P-R curve")
    plt.xlabel("召回率 recall")
    plt.ylabel("精准率 precision")
    # plt.show()

    plot_precision_recall_curve(LR, X_test_std, y_test)
    plt.show()
    # plt.title("Precision Recall vs Threshold Chart")
    # plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    # plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    # plt.ylabel("Precision, Recall")
    # plt.xlabel("Threshold")
    # plt.legend(loc="lower left")
    # plt.ylim([0,1])
    # plt.show()


    # ### 讨论logistic regression 阈值问题;上面使用了 precision_recall_curve()方法 https://www.codingdict.com/sources/py/sklearn.metrics/5934.html
    # prob_array = np.array(lr_probs)
    # thresholds_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # j = 1
    # for threshold in thresholds_list:
    #     pred_y_new = np.zeros([len(y_test), 1])
    #     pred_y_new[prob_array[:, 1] > threshold] = 1
    #     ## 获得混淆矩阵
    #     plt.subplot(3, 3, j)
    #     conf = confusion_matrix(y_test, pred_y_new)
    #     ## 画图
    #     accurracy = (conf[0, 0] + conf[1, 1]) / (conf[0, 0] + conf[0, 1] + conf[1, 0] + conf[1, 1])
    #     # 召回率
    #     recall = conf[1, 1] / (conf[1, 0] + conf[1, 1])
    #     # 精准率
    #     recall = conf[1, 1] / (conf[0, 1] + conf[1, 1])
    #     j = j + 1

def logistic_regression_cv_breast_cancer():

    canver = load_breast_cancer()
    X = canver.data
    y = canver.target
    print("数据规模：{}".format(X.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=0)
    sc = StandardScaler() # #标准化数据，保证每个维度的特征数据方差为1，均值为0.使得预测结果不会被某些维度过大的特征值主导
    x_train_std = sc.fit_transform(X_train) # #先拟合，在转换
    X_test_std = sc.transform(X_test) ##上面拟合过，这里直接转换

    #构建并训练模型
    ##  multi_class:分类方式选择参数，有"ovr(默认)"和"multinomial"两个值可选择，在二元逻辑回归中无区别
    ##  cv:几折交叉验证
    ##  solver:优化算法选择参数，当penalty为"l1"时，参数只能是"liblinear(坐标轴下降法)"
    ##  "lbfgs"和"cg"都是关于目标函数的二阶泰勒展开
    ##  当penalty为"l2"时，参数可以是"lbfgs(拟牛顿法)","newton_cg(牛顿法变种)","seg(minibactch随机平均梯度下降)"
    ##  维度<10000时，选择"lbfgs"法，维度>10000时，选择"cs"法比较好，显卡计算的时候，lbfgs"和"cs"都比"seg"快
    ##  penalty:正则化选择参数，用于解决过拟合，可选"l1","l2"
    ##  tol:当目标函数下降到该值是就停止，叫：容忍度，防止计算的过多
    lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=2,penalty="l2",solver="lbfgs",tol=0.01)
    re = lr.fit(x_train_std,y_train)
    #模型效果获取
    r = lr.score(X_test_std,y_test)
    print("R值(准确率):",r)
    print("参数:",re.coef_)
    print("截距:",re.intercept_)
    print("稀疏化特征比率:%.2f%%" %(np.mean(lr.coef_.ravel()==0)*100))
    print("=========sigmoid函数转化的值，即：概率p=========")
    print(re.predict_proba(X_test))     #sigmoid函数转化的值，即：概率p

    import joblib
    joblib.dump(sc, "save/logistic_sc.model")
    joblib.dump(lr, "save/logistic_lr.model")
    joblib.load("save/logistic_sc.model")
    joblib.load("save/logistic_lr.model")

    # 预测
    X_test = sc.transform(X_test)  # 数据标准化
    y_predict = lr.predict(X_test)

    x = range(len(X_test))
    plt.figure(figsize=(14, 7), facecolor="w")
    plt.ylim(0, 6)
    plt.plot(x, y_test, "ro", markersize=8, zorder=3, label=u"真实值")
    plt.plot(x, y_predict, "go", markersize=14, zorder=2, label=u"预测值,$R^2$=%.3f" % lr.score(X_test, y_test))
    plt.legend(loc="upper left")
    plt.xlabel(u"数据编号", fontsize=18)
    plt.ylabel(u"乳癌类型", fontsize=18)
    plt.title(u"Logistic算法对数据进行分类", fontsize=20)
    plt.savefig("Logistic算法对数据进行分类.png")
    plt.show()

    print("=============Y_test==============")
    print(y_test.ravel())
    print("============Y_predict============")
    print(y_predict)



def linear_regression_diabetes():
    """
    在该数据集中，包括442个病人的生理数据及一年以后的病情发展情况。
    数据集中的特征值总共10项：年龄、性别、体质指数、血压、s1~s6（6种血清的化验数据）。但需要注意的，以上的数据是经过预处理， 10个特征都做了归一化处理。
    第11项数据，是我们的要预测的目标值，一年疾后的病情定量测量，它是一个连续的实数值，符合线性回归模型评估的范畴。
    :return:
    """
    # (1)导入数据
    diabetes = load_diabetes()
    X, y = load_diabetes().data, load_diabetes().target
    print(diabetes.keys())
    # dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])

    # （2）分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # （3）训练
    LR = LinearRegression()
    performance  = LR.fit(X_train, y_train)

    # （4）预测(本例分必要)
    y_pred_train = LR.predict(X_train)  # 在测试集合上预测
    y_pred_test = LR.predict(X_test)  # 在测试集合上预测

    #(5) 评估模型
    print("训练集合上R^2 = {:.3f}".format(performance.score(X_train, y_train)))
    print("测试集合上R^2 = {:.3f} ".format(performance.score(X_test, y_test)))
    print("训练集合上R^2 = {:.3f}".format(LR.score(X_train, y_train)))
    print("测试集合上R^2 = {:.3f} ".format(LR.score(X_test, y_test)))
    print("训练集合上R^2 = {:.3f}".format(r2_score(y_train, y_pred_train)))
    print("测试集合上R^2 = {:.3f} ".format(r2_score(y_test,y_pred_test)))
    """
    训练集合上R^2 = 0.555   ;测试集合上R^2 = 0.359
    训练集上R2大于测试集，这是符合预期的；如果两者差距过大，说明存在一定的过拟合；
    R2->1越好，反之表示模型的数据拟合度越差；
 
    """

    np.set_printoptions(precision = 3, suppress = True)
    print('w0 = {0:.3f}'.format(LR.intercept_))
    print('W = {}'.format(LR.coef_))

if __name__ == "__main__":
    # kneighbors()

    # svcl()
    # logistic_regression_breast_cancer()
    # https: // blog.csdn.net / loveliuzz / article / details / 78708359
    logistic_regression_cv_breast_cancer()

    # linear_regression_diabetes()

