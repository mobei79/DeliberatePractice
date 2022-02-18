# -*- coding: utf-8 -*-
"""
@Time     :2021/7/9 17:12
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import tensorflow as tf
import numpy as np

# create some data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# create TensorFlow structure start #
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0)) # 定义tf的参数变量，这里使用随机数列生成； 以为 -1 到 1
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data)) #计算差，最初会有很大的差别，需要不断优化
optimizer = tf.gradients(0.5) # 建立一个优化器 。学习效率0.5
train = optimizer.minimize(loss)   # 用优化器减少误差

init = tf.initializers()#
# create TensorFlow structure end #

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step/20 == 0:
        print(step,sess.run(Weights),sess.run(biases))