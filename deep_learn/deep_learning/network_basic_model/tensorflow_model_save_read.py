# -*- coding: utf-8 -*-
"""
@Time     :2021/9/3 14:49
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""


# v1 = tf.Variable(<initial - value>，name=<optional - name>) # 此函数用于定义图变量。生成一个初始值为initial - value的变量
import tensorflow as tf

v1 = tf.get_variable("v1", shape=[1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0), dtype=tf.float32)
v2 = tf.get_variable("v2", shape=[1], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0), dtype=tf.float32)
result = v1 + v2

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "model.ckpt")
