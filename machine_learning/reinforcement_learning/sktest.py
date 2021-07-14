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
X = np.array(range(20))
kf = KFold(n_splits=2,shuffle=True,random_state=2)
print(kf.get_n_splits())

print(X)
print(kf.split(X))
for x, y in kf.split(X):
    print(len(x),x)
    print(len(y),y)
    print("*"*13)