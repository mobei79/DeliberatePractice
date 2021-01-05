# -*- coding: utf-8 -*-
"""
@Time     :2020/11/10 15:33
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import datetime

# from sklearn import tree
# feature = [[131,1],[12324,1],[231,0],[1231,0]]
# label = ['M','M','F','F']
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(feature,label)
# print(clf.predict([[122,0]]))
import os
import sys
lib_path = os.path.abspath(os.curdir+'/..')
sys.path.append(lib_path)
print(lib_path)
# lib_path = os.path.abspath(os.curdir + '/../../../sbin')
# sys.path.append(lib_path)
from HiveTask import HiveTask
# HiveTask(sys.argv[1:])
print(sys.argv[1:])
HiveTask([1,2,3,4,5])
ht = HiveTask(sys.argv)
print(ht.partition_dt)

def getDateDelta(basetime,delta):
    # basetime is a string like 2020-11-11 15:10:51
    time = datetime.datetime(basetime[0:4],basetime[5:7],basetime[8:10])
    print(time)

# getDateDelta("2020-11-11 15:14:46",'23')


def __random_file_name():
    import random
    import string
    tmp = ''.join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(10))
    tmp_file = '/tmp/' + tmp + '.dat'
    return tmp_file
print(__random_file_name())