# -*- coding: utf-8 -*-
"""
@Time     :2021/9/3 10:11
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc : https://zhuanlan.zhihu.com/p/296379158
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import gfile
import time # time库用来获取当下时间戳，为绘以时间作为自变量的图象准备
import os # s 用来生成记录训练日志的文件夹
import datetime # 对生成的训练日志按时间命名

'''
定义要拟合的函数，带高斯噪音的x_data, y_data 作为训练集；不带噪音的作为验证集；
训练集60 验证集20 测试集20；
'''
def function():
    x_data = np.linspace(-1, 30, 3000) # 构建等差数列
    noise = np.random.normal(0, 0.05, x_data.shape) # 正态分布 （loc均值， scale标准差， sizes输出值规模）
    # y_data = 5 * np.exp(-x_data/4.56) + noise
    y_data = np.sin(x_data) + noise + 5 * np.exp(-x_data/4.56)
    x_val_data = np.linspace(10, 20, 1000)
    y_val_data = np.sin(x_val_data) + 5 * np.exp(-x_val_data/4.56)
    return x_data, y_data, x_val_data, y_val_data

"""
创建全连接模型
tf.keras.Sequential() 可以快速简洁的封装网络层。
tf.keras.layers.Flatten(input_shape=(1,)) 在数据输入进全连接层时，要对数据进行铺平处理,
    输入数据shape 为（1，）因为输入的为[1., 2., ..... ]形式的张量。
tf.keras.layers.Dense(100, activation='relu'), 设置全连接层，神经元数量为100，激活函数为‘relu’函数。
tf.keras.layers.Dropout(0.2), 设置dropout ，防止过拟合
tf.keras.layers.Dense(1) 设置一个神经元，输出预测的y列表

optimizer = tf.keras.optimizers.Adam(0.01) 设置adam作为优化器，学习率0.01
model.compile(optimizer=optimizer, loss="mse")编译模型，优化目标为“mse”使均方差函数最小
"""
def build_model():
    model = tf.keras.Sequential(# tf.keras.Sequential() 可以快速简洁的封装网络层。
        [
            tf.keras.layers.Flatten(input_shape=(1,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            # tf.keras.layers.Dense(100, activation='relu'),
            # tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ]
    )
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=optimizer, loss="mse")
    return model


if __name__ == "__main__":
    time1 = time.time()
    timex = []
    loss = []
    val_loss = []
    # graph = tf.compat.v1.get_default_graph()
    # graphdef = graph.as_graph_def()

    x_data, y_data, x_val_data, y_val_data = function()
    model = build_model() # model = build_model() 调用函数，建立网络模型

    log_dir = log_dir = 'D:\\tmp\\tf_fit_logdir'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(log_dir,) # 把日志写在log_dir 路径内。
    # model.fit(x_data, y_data, batch_size=256, epochs=5, validation_data=(x_val_data, y_val_data))



    # 11放测试数据；21放样本书数据；
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    bx = fig.add_subplot(2, 2, 2)
    cx = fig.add_subplot(2, 2, 3)
    dx = fig.add_subplot(2, 2, 4)
    ax.plot(x_data, y_data)
    cx.plot(x_val_data, y_val_data)
    plt.ion()
    plt.show()

    for i in range(2000):
        # model 执行模型的训练操作;
        #   batch 包含N个样本的结合,每一个batch的样本都是独立并行处理的,在训练时，一个 batch 的结果只会用来更新一次模型。
        #       一个batch的样本通常比单个输入更能接近总体输入数据的分布,在预测时,建议选择尽可能大的batch;
        #   epochs 是训练的轮次，fit一次模型，就训练epochs次；
        hist = model.fit(x_data, y_data, batch_size=256, epochs=5, validation_data=(x_val_data, y_val_data))
        timex.append(time.time() - time1)   # 记录训练时间
        loss.append(hist.history['loss'])   # 记录训练误差
        val_loss.append(hist.history['val_loss'])   # 记录验证误差：hist.history 返回的是一个字典型数据
        print("model fit loss=")
        print(hist.history['loss'])
        print("model fit val_loss=")
        print(hist.history['val_loss'])

        # 绘制训练误差曲线，验证误差曲线。每次5步会出现这5次的误差，可以对他们进行一定处理后输出，为了简单我就只取了每5次误差的第一个值作为图像的输出。这两条语句的图像输出需要调用tensorboard查看。
        with summary_writer.as_default():
            tf.summary.scalar('train-loss', float(hist.history['loss'][0]), step=i)
            tf.summary.scalar('val-loss', float(hist.history['val_loss'][0]), step=i)
        if i % 2 == 0:
            try:
                ax.lines.remove(lines[0])
                bx.lines.remove(lines2[0])
                cx.lines.remove(lines4[0])
                dx.lines.remove(lines3[0])
            except Exception:
                pass

            y_pred = model.predict(x_data)
            lines4 = cx.plot(x_val_data, model.predict(x_val_data)) # 绘制验证集图像拟合效果
            lines3 = dx.plot(timex, val_loss)   # 验证误差
            lines2 = bx.plot(timex, loss)   # 训练误差
            lines = ax.plot(x_data, y_pred) # 练图像效果
            plt.pause(0.0001) # 因为可能由于迭代速度过快，图像显示不出来，设置了一个很小的停顿
        """
        最后设置一个停止训练的条件，为了方便我就写了一个取五次验证误差最大值小于0.0003为终止的条件，可能拟合效果达不到0.0003的要求，可以根据要求调整终止的条件。
        加入了一行模型保存的语句示例，具体保存可以设置保存权重或者网络结构，或者整个网络的保存，代码中的是全模型保存。
        """
        if tf.reduce_max(hist.history['val_loss']) < 0.0003:
            model.save(filepath="D:\\tmp\\keras_network_savedModel")
            break
    model.save(filepath="")
    plt.pause(10)