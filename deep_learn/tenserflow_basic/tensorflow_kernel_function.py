# -*- coding: utf-8 -*-
"""
@Time     :2022/2/24 10:26
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import tensorflow as tf
import numpy as np
print(tf.__version__)

"""
tf.data.Dataset.from_tensor_slices((x, y))
    Dataset的核心函数之一，把给定的元组、列表和张量等数据进行特征切片。切片的范围从在外层维度开始。如果多个特征进行组合，一次切片就是把每个组合的最外层维度的数据切开，分成一组一组的。
    eg. 我们两组数据，分别是特征和标签，为了简化说明问题，假设三个特征对应一个标签，我们把特征和标签组合成一个tuple。
        这时我们想让每个标签恰好对应三个特征，就像直接切片，如:[f11,f12,f13][l1]。from_tensor_slices()就能实现，
"""
# 构建一个6*3维度feature矩阵，表示6个样本，每个样本特征维度为3；
features, labels = (#np.array([["guo"],["jing"],["jin"],["jin"],["jing"],["guo"]]),
                    np.random.sample((6, 4)),
                    np.random.sample((6, 1)))
print("***** show features  and labels:\n",features,labels)
print(type(features))
print(type((features,labels)))

# 输出的类型： <TensorSliceDataset shapes: ((3,), (1,)), types: (tf.string, tf.float64)>;
# 属于tf中的TensorSliceDataset类，shape就是切片后一一对应的维度;
data = tf.data.Dataset.from_tensor_slices((features, labels))
print("data ofter from_tensor_slices:\n", data)
print(type(data))

"""
shuffle(
    buffer_size, 元素个数，最完美的shuffle是所有数据一起shuffle，但是为了避免内存不足，每次选buffer_size个数据进行
    seed=None,
    reshuffle_each_iteration=None
)
eg. Dataset 有100000个元素，buffer_size=1000,那么首先会对前1k个数据进行。
"""

"""

tf.train.batch(
    tensors,
    batch_size,
    num_threads=1,
    capacity=32,
    enqueue_many=False,
    shapes=None,
    dynamic_pad=False,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None

————————————————
函数功能：利用一个tensor的列表或字典来获取一个batch数据

参数介绍：
函数功能：利用一个tensor的列表或字典来获取一个batch数据

参数介绍：

tensors：一个列表或字典的tensor用来进行入队
batch_size：设置每次从队列中获取出队数据的数量
num_threads：用来控制入队tensors线程的数量，如果num_threads大于1，则batch操作将是非确定性的，输出的batch可能会乱序
capacity：一个整数，用来设置队列中元素的最大数量
enqueue_many：在tensors中的tensor是否是单个样本
shapes：可选，每个样本的shape，默认是tensors的shape
dynamic_pad：Boolean值.允许输入变量的shape，出队后会自动填补维度，来保持与batch内的shapes相同
allow_samller_final_batch：可选，Boolean值，如果为True队列中的样本数量小于batch_size时，出队的数量会以最终遗留下来的样本进行出队，如果为Flalse，小于batch_size的样本不会做出队处理
shared_name：可选，通过设置该参数，可以对多个会话共享队列
name：可选，操作的名字
————————————————
版权声明：本文为CSDN博主「修炼之路」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/sinat_29957455/article/details/83152823
"""



"""
查看tensorflow版本
"""
import os
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

"""
定义训练环境
"""
use_cuda = False #在cpu上训练