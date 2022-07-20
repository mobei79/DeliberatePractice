# -*- coding: utf-8 -*-
"""
@Time     :2022/3/27 15:14
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import numpy as np
import math,random
def GD(epoches, weight, x_train, y_train, learning_rate):
    for epoch in range(epoches):
        y_pred = np.dot(weight, x_train.T)
        deviation = y_pred - y_train.reshape(y_pred.shape)
        gradient = 1/len(x_train)*np.dot(deviation, x_train)
        weight = weight - learning_rate*gradient

def SGD(epoches, weight, x_train, y_train, learning_rate):
    for epoch in range(epoches):
        for i in range(len(x_train)):
            index = random.randint(0, len(x_train))
            y_pred = np.dot(weight, x_train[index].T)
            deviation = y_pred - y_train[index].reshape(y_pred.shape)
            gradient = np.dot(deviation, x_train[index])
            weight = weight - learning_rate*gradient

 """
        Permutation()函数的意思的打乱原来数据中元素的顺序。
        输入为整数，返回一个打乱顺序的数组
        输入为数组/list，返回顺序打乱的数组/list
        与Shuffle()的区别：
        Shuffle()在原有数据的基础上操作，打乱元素的顺序，无返回值
        Permutation,不是在原有数据的基础上操作，而是返回一个新的打乱顺序的数组
        """
def MBGD(epoches, weight, x_train, y_train, learning_rate, batch_size):
    def batch_generator(x, y, batch_size):
        n_samples = len(x)
        batch_num = int(n_samples/batch_size)
        indexes = np.random.permutation(n_samples)
        for i in range(batch_num):
            yield (x[indexes[i*batch_size:(i+1)*batch_size]],
                   y[indexes[i*batch_size:(i+1)*batch_size]])

    for epoch in range(epoches):
        for x_batch, y_batch in batch_generator(x_train, y_train, batch_size):
            y_hat = np.dot(weight, x_batch.T)
            deviation = y_hat - y_batch.reshape(y_hat.shape)
            gradient = 1/len(x_batch)*np.dot(deviation, x_batch)
            weight = weight-learning_rate*gradient