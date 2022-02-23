# -*- coding: utf-8 -*-
"""
@Time     :2022/2/15 13:29
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

"""
准备训练数据
1 下载数据
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
2 下载好的图处理成可训练的数据
    ImageDateGenerator 
    图片大小不一；数据量大不能一次性装入内存；准备训练数据时经常需要修改参数，例如输出的尺寸（150或者换成300），增补图像拉伸（镜像 旋转 拉伸来增强图像效果）等
"""
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255) #scale把数值压缩在0-1
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory("D:\Program Files\pythonwork\dataset\DeliberatePractice\deep_learn\\validation-horse-or-human-mini/",
                                                    target_size=(300,300),  # 输出的size
                                                    batch_size=32,          # 每一批是多少
                                                    class_mode='binary')    # 二分类
validation_generator = train_datagen.flow_from_directory("D:\Program Files\pythonwork\dataset\DeliberatePractice\deep_learn\\horse-or-human-mini/",
                                                    target_size=(300,300),
                                                    batch_size=32,
                                                    class_mode='binary')
# Found 24 images belonging to 2 classes.
# Found 24 images belonging to 2 classes.

"""
构建模型
    注：图像向后经过三次卷积 16,32,64。根据过程推想是：一张图片经过卷积池化为16个图片；然后卷积池化为32张图片；然后卷积池化为64张特征图片；
    最后将图片展平用于分类；
"""

# model = keras.Sequential([])
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)), #(None, 298, 298, 16) 448=(3*3*3+1)*16  448
    tf.keras.layers.MaxPooling2D(2, 2), # (None, 149, 149, 16)
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),  #(None, 147, 147, 32) 4640=(3*3*16+1)*32
    tf.keras.layers.MaxPooling2D(2, 2), #(None, 73, 73, 32)
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  #(None, 71, 71, 64)  18496= (3*3*32+1)*64
    # tf.keras.layers.MaxPooling2D(2, 2), #（None，35，35，64）
    tf.keras.layers.Flatten(),  # （None, 78400）
    tf.keras.layers.Dense(512, activation='relu'),  #(None, 512)  4014312 = (78400 + 1)*512
    keras.layers.Dense(1, activation='sigmoid')     # 二分类问题， 输出0或者1 (None, 1) 513个参数
])
model.summary()
model.compile(loss=tf.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['acc'])


"""
训练模型
"""

model.fit(
    train_generator, # 原来的输入
    epochs=5,
    verbose=1,
    validation_data= validation_generator, # validation test部分
    validation_steps=8
)

"""
优化参数
池化中的filter放多少个？
conv2D层数？是两层还是三层还是一层；
dense层的神经元是多少个？
learning rate? (RMSprop())
方法1： 手工调整，或者写个循环并记录下来；
方法2：使用kerasTumer库来优化
"""
# from kerastuner.tuners impot Hyperband
# from kerastuner.engine.hyperparameters import HyperParameters
# hp = HyperParameters()
# def build_model(hp):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(hp.Choice('num_filters_layer0',values[16, 64],default=16), (3, 3), activation='relu', input_shape=(300, 300, 3)),
#         # hp.Choice('num_filters_layer0',values[16, 64],default=16) 给第一层filter参数加一个名字，和范围，会将参数保存到这个名称中
#         # (None, 298, 298, 16) 448=(3*3+1)*16  448
#         tf.keras.layers.MaxPooling2D(2, 2),
#         for i in range(hp.Int("num_conv_layers",1,3,)): # 卷积层层数过大会导致图片变小无法预测
#               # (None, 149, 149, 16)
#             tf.keras.layers.Conv2D(hp.Choice('num_filters_layer1',values[16, 64],default=16), (3, 3), activation='relu', input_shape=(300, 300, 3)),
#             tf.keras.layers.MaxPooling2D(2, 2),  # (None, 73, 73, 32)
#         # (None, 147, 147, 32) 4640=(3*3*16+1)*32
#
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
#         # (None, 71, 71, 64)  18496= (3*3*32+1)*64
#         tf.keras.layers.MaxPooling2D(2, 2),  # （None，35，35，64）
#         tf.keras.layers.Flatten(),  # （None, 78400）
#         tf.keras.layers.Dense(hp.Int("hidden_units",128,512,step=32), activation='relu'),  # (None, 512)  4014312 = (78400 + 1)*512
#         keras.layers.Dense(1, activation='sigmoid')  # 二分类问题， 输出0或者1 (None, 1) 513个参数
#     ])
#     model.compile(loss=tf.losses.binary_crossentropy,
#                   optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
#                   metrics=['acc'])
#     return model
# tuner = HyperBand(build_model,
#                   objective='val_acc', # 以测试集的准确李为标准
#                   max_epoch=15,
#                   directory="/logs/house_human_params",
#                   hyperparameter=hp,
#                   project_name="My_house_human_project")
# # 搜索参数
# tuner.search(train_generator, epochs=10, validation_data=validation_generator)
# #找到最好的参数
# best_hps = tuner.get_best_hyperparameters(1)[0]
# best_hps.values
# #根据最优的参数将模型构建起来
# model = tuner.hypermodel.build(best_hps)
# model.summary()