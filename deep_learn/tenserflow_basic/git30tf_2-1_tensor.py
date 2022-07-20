# -*- coding: utf-8 -*-
"""
@Time     :2022/2/28 16:23
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

"""
常量张量
"""
import numpy as np
import tensorflow as tf

i = tf.constant(1) # tf.int32 类型常量
l = tf.constant(1, dtype=tf.int64) # tf.int64 类型常量
f = tf.constant(1.23) # tf.float32 类型常量
d = tf.constant(3.23, dtype=tf.double) #tf.double 类型常量
s = tf.constant("hello world") # tf.string 类型常量
b = tf.constant(True) # tf.bool 类型常量

print(tf.int64 == np.int64)
print(tf.bool == np.bool)
print(tf.double == np.float64)
print(tf.string == np.unicode) # tf.string类型和np.unicode类型不等价

"""
不同类型的数据可以用不同维度(rank)的张量来表示。

标量为0维张量，向量为1维张量，矩阵为2维张量。

彩色图像有rgb三个通道，可以表示为3维张量。

视频还有时间维，可以表示为4维张量。

可以简单地总结为：有几层中括号，就是多少维的张量。
"""
scalar = tf.constant(True)
print(tf.rank(scalar))
print(scalar.numpy().ndim)