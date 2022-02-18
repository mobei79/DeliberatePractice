# -*- coding: utf-8 -*-
"""
@Time     :2022/2/15 12:42
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import numpy as np
from tensorflow import keras
import tensorflow as tf
"""
卷积：convolution
    卷积核\过滤器：不同的过滤器有不同的效果，如竖线、横线等等
池化：max pooling
    
通过卷积和池化得到的图像更小，减少了数据，增强特征

卷积网络的结构解析：
    一共七层
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 64)        640       # 输入28*28，过滤器3*3，过滤掉两个像素，变为26*26,64为64个过滤器，变成了64张图片
_________________________________________________________________# （3*3+1bias）*64 = 640，过滤器有3*3九个参数，加1个bias，再乘以64个过滤器
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0         # 长宽各减半，
_________________________________________________________________# 没有参数
conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928     # 再次去掉两个像素
_________________________________________________________________# （3*3*64+1）*64 上面的卷积池化变成了64张图片，每个图片都需要3*3的卷积层参数+1个bias
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         # 
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         # 将数据展平 5*5*64个元素
_________________________________________________________________
dense (Dense)                (None, 128)               204928    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 243,786
Trainable params: 243,786
Non-trainable params: 0
_________________________________________________________________
Epoch 1/5

查看每一层的输出?
    
"""
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()


# 构建模型，CNN在全连接神经网络基础上增加了4层
model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    # 两维的卷积层，64为64个过滤器，每个过滤器3*3，输入为28*28*1 只有一个灰度值
    keras.layers.MaxPooling2D(2,2), # 2*2 的池化层 横竖各减少一半
    keras.layers.Conv2D(64, (3,3), activation='relu'), # 不用再指定输入维度了
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(), # 中间就不用指定维度了
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.summary()


"""
之后和全连接DNN一样， 只不过训练时间更长
报错ValueError: Input 0 of layer sequential is incompatible with the layer: expected ndim=4, found ndim=3. Full shape received: [32, 28, 28]
    输入数据的dimension问题
        train_images_scaled.reshape(-1,28,28,1)
    ps：感觉是CNN加了卷积和池化之后就会出现这个情况
"""
# 模型训练   为了训练效果更好，训练数据要normalization标准化 或者scaling归一化
train_images_scaled = train_images / 255
# model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,  # 多类别变量使用 sparse；如果是onehot类别时 就用categorical_cro
              metrics=['accuracy'])  # 训练过程中想要看到精度
model.fit(train_images_scaled.reshape(-1,28,28,1),
          train_labels,
          epochs=1)

# # 用test数据评估模型准确度
# test_images = test_images / 255  # 如果训练样本归一化了，训练样本也需要归一化
# model.evaluate(test_images.reshape(-1,28,28,1), test_labels)
#
# # 判断单张类别 ，输出的为10种类别中各自的概率值。
# model.predict([[test_images[0] / 255]])  # 加中括号是为了满足对输入的要求，同时需要归一化
# np.argmax(model.predict([[test_images[0] / 255]]))
# print(test_labels[0])

# 查看每一层的输出
layer_outputs = [layer.output for layer in  model.layers]
activation_model = tf.keras.models.Model(inputs = model.inputs, outputs = layer_outputs)
pred = activation_model.predict(test_images[0].reshape(1,28,28,1))
pred#.shape # 7层，每层的输出
pred[0].shape # 卷积层的shape （1,26,26,64）
pred[0][0, :,:,1] # 取第二个过滤器的卷积结果