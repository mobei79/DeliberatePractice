# -*- coding: utf-8 -*-
"""
@Time     :2020/11/19 16:44
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import sklearn

print("232")
datasets = {'banala':{'long':400,'not_long':100,'sweet':350,'not_sweet':150,'yellow':450,'not_yellow':50},
            'orange':{'long':0,'not_long':300,'sweet':150,'not_sweet':150,'yellow':300,'not_yellow':0},
            'other_fruit':{'long':100,'not_long':100,'sweet':150,'not_sweet':50,'yellow':50,'not_yellow':150}}

for i in datasets:
    print(22)
    print(i)

def count_total(data):
    """
    计算各种水果的总数
    :param data:
    :return:
    """
    count = {}
    total = 0
    for fruit in data:
        count[fruit] = data[fruit]['sweet'] + data[fruit]['not_sweet']
        total += count[fruit]
    return count,total
