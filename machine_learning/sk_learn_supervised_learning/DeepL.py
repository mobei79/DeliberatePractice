# -*- coding: utf-8 -*-
"""
@Time     :2022/2/14 17:01
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import numpy as np
from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences

from torch.optim import Optimizer   #Pytorch中优化器接口
from torch import nn                #Pytorch中神经网络模块化接口'
# class XXmodel(nn.Module):
    # def forward(self, input):