# -*- coding: utf-8 -*-
"""
@Time     :2022/2/15 11:04
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
拟合函数y = 2x-1
只用一个神经元来拟合函数
"""
from tensorflow import keras
"""
keras 是tenserflow中一个高级api 使用起来比较简单
通常分为如下几步
"""

# 构建模型  : 一个layer表示一层，units=1表示只有一个神经元
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model.fit(xs, ys, epochs=100)  # 所有的数据需要跑500次。

print(model.predict([10.0]))
