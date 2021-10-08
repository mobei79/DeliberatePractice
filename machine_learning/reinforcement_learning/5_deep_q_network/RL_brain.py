# -*- coding: utf-8 -*-
"""
@Time     :2021/9/3 9:42
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.random_normal_initializer(seed=1)

class DeepQNetWork:
    def __init__(
            self,n_actions,n_features,learning_rate=0.01,
    reward_decay=0.9,e_greedy=0.9,replace_target_iter=300,
    memory_size=500,
    batch_size=32,
    e_greedy_increment=None,
    output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()