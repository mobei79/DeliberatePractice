# -*- coding: utf-8 -*-
"""
@Time     :2021/7/12 15:29
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import numpy as np
from sklearn.model_selection import KFold


# X = np.array(range(20))
# kf = KFold(n_splits=2,shuffle=True,random_state=2)
# print(kf.get_n_splits())
#
# print(X)
# print(kf.split(X))
# for x, y in kf.split(X):
#     print(len(x),x)
#     print(len(y),y)
#     print("*"*13)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

y_true = [0, 0, 1, 1]
y_score = [0.1, 0.4, 0.35, 0.8]

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
print(precision)
print(recall)
print(thresholds)
"""
[0.66666667 0.5 1. 1.]
[1.  0.5 0.5 0. ]
[0.35 0.4  0.8 ]
"""
plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0, 1])
plt.show()