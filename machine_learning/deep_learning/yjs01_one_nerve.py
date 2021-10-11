# -*- coding: utf-8 -*-
"""
@Time     :2021/10/11 11:01
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import tensorflow as tf
print(tf.__version__)
import numpy as np
"""
x = -1, 0,1,2,3,4
y = -3,-1,1,3,5,7
y = 2x-1
"""

from tensorflow import  keras

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])     #keras 基础的模型都用Sequential
model.compile(optimizer='sgd', loss='mean_squared_error')  # s

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
