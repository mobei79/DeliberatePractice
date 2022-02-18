# -*- coding: utf-8 -*-
"""
@Time     :2022/2/18 10:00
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
"""
这是TensorFlow2.0 简明学习课程的代码实现：
目的：简单了解tf2.0使用方法和特点，算是简单的入门；后续需要进一步了解在实现过程中要注意的核心点。
课程地址：https://www.bilibili.com/video/BV1Zt411T7zE?p=4&spm_id_from=333.1007.top_right_bar_window_history.content.click
"""

"""
output Shape:(None, 1)  第一个维度：None表示样本的维度，在构建模型的时候不需要考虑；第二个维度表示输出的维度；
优化器（optimizer）是编译模型所需要的两个参数之一（另一个是loss），1可以先实例化一个优化器对象，然后将它传入model.compile()；2，在compile中通过名称直接调用优化器(此时优化器将使用默认参数)。
    常见优化函数：
        SGD（抽取m个小批量样本，计算平均梯度值。参数：lr学习速率 momentum[用于加速SGD在相关方向上前进并抑制震荡]，decay每次参数更新后学习率衰减值，nesterov）
        RMSprop：经常用于处理序列问题（文本分类等）
        Adam：最常用的优化器，可以看做修正后的Momentum+RMSprop算法，对超参数的选择相当鲁棒不敏感，学习速率建议是0.001；
            Adam是一种可以替代SGD的一阶优化算法，基于训练数据迭代更新权重；通过计算梯度的一阶矩估计和二阶矩估计，而为不同的参数独立的自适应性学习率

函数式API：把每一层看做一个函数来调用，（相较于Sequential是单输入单输出中间所有层是顺序连接的，这种结构比较单一。不方便引入残差或者将输入直接连接到输出），就可以自定义网络结构。
    
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images/255
test_images = test_images/255

# 函数式API的核心在于可以调用，需要那一层，就直接调用每一层；Sequential模型需要先建立一个模型
input = keras.Input(shape=(28,28)) # 第一维是样本维度，不需要设定；只需要设定样本本身的维度即可。

x = keras.layers.Flatten()(input) # 这里的input就是参数，即调用输入层； keras.layers.Flatten()是类对象，后面的(input)实质上是调用了类方法call()
x = keras.layers.Dense(32, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation="relu")(x)
output = keras.layers.Dense(10, activation="softmax")(x)
# 上面就建立好每一层了
model = keras.Model(input, output) # 告诉Model input和output。根据两者建立好模型
model.compile(optimizer="adam", loss=tf.losses.sparse_categorical_crossentropy,
              metrics=["acc"])

history = model.fit(train_images,
                    train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels)
                    )
# 多输入多输出指的是：假设我们有两份数据设定input1和input2,要判断两类图片是不是一样，
input1 = keras.Input(shape=(28,28)) #
input2 = keras.Input(shape=(28,28)) #
x1 = keras.layers.Flatten()(input1)
x2 = keras.layers.Flatten()(input2)
x_new = keras.layers.concatenate([x1, x2])
output_new = keras.layers.Dense(1, activation="sigmoid")(x_new)
model = keras.Model(input=[input1,input2], output=output_new)
# 在这里可以指定两个输入，他们分别进行Flatten，然后合并，进入下面的Dense