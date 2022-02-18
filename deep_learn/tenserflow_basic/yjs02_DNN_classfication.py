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
from tensorflow import keras

print(tf.__version__)
import numpy as np

"""
数据集
    10万张图片；10各类别；28*28像素；每个像素是一个灰度值（0,255）；
    用数据和标签做训练，得到一个神经网络模型

"""

# 加载数据集 使用load_data() 加载后分为了四类。
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 观察数据
# print(train_images[0])
# print(train_images.shape) # (60000, 28, 28) 6万张，每张28*28
# print(train_labels[:10])
# import matplotlib.pyplot as plt
# plt.imshow(train_images[0])
# plt.show()

# 自动终止训练
"""
训练次数如果太多，很容易过拟合（训练loss很小但是测试loss较大）
所以训练次数不是越多越好；
可以通过fit中的callbacks参数来控制是否后总值训练。
    我们可以自定义myCallback类，继承Callbcak； 在自定义类中替换这些方法，
"""


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get("loss") < 0.4):
            print("\n loss is low so cancelling training!!")
            self.model.stop_training = True


callbacks = myCallback()

"""
神经元就是输入加权累加，再放入激活函数中，得到神经元输出

 构造模型 这里用三层结构，第一次接收输入（28*28）；中间层自己定义； 最后一层为输出（10）
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 输入层 将28*28展平为784个像素
    keras.layers.Dense(128, activation=tf.nn.relu),  # 128个神经元
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 10个神经元
])

model.summary()  # 使用summary 查看模型的构造
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
Process finished with exit code 0

100480 为：784个像素输入*128个神经元，每个像素都给到每个神经元， 784*128=100352 + 128（bias）
1290 为：输出层10个神经元，128+1个输入 
输出层没有bias
"""

# 为了训练效果更好，训练数据要normalization标准化 或者scaling归一化
train_images = train_images / 255
# model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,  # 多类别变量使用 sparse；如果是onehot类别时 就用categorical_cro
              metrics=['accuracy'])  # 训练过程中想要看到精度

model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
# 用test数据评估模型准确度
test_images = test_images / 255  # 如果训练样本归一化了，训练样本也需要归一化
model.evaluate(test_images, test_labels)
# 判断单张类别 ，输出的为10种类别中各自的概率值。
model.predict([[test_images[0] / 255]])  # 加中括号是为了满足对输入的要求，同时需要归一化
np.argmax(model.predict([[test_images[0] / 255]]))
print(test_labels[0])



