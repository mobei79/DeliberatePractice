# -*- coding: utf-8 -*-
"""
@Time     :2020/11/9 18:04
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import os,pandas
def dubble_sort(numList,orderby=1):
    if not numList:
        print("the input is null")
        return []
    print(orderby)
    for i in range(len(numList)-1):
        ex_flag = False # 交换标志位，没有发生交换则说明排序成功
        for j in range(len(numList)-i-1):
            if numList[j]>numList[j+1]:
                ex_flag = True
                numList[j],numList[j+1]=numList[j+1],numList[j]
                print(i,j,numList)
        if not ex_flag:
            return numList
    return numList


def fake_bubble(num):  # 这个是伪冒泡排序！！！
    count = len(num)
    for i in range(count):
        for j in range(i+1, count):
            if num[i] > num[j]:
                num[i], num[j] = num[j], num[i]
                print(i, j, num)
    return num
rst = dubble_sort([14,12,11,15,9,8,7,6,5])
# fake_bubble([14,12,11,15,9,8,7,6,5])
print(rst)