# -*- coding: utf-8 -*-
"""
@Time     :2021/7/20 10:53
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    # print (x_train[0])

    x_train = x_train / 255
    x_test = x_test / 255
    print(x_train)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()
model = Sequential()
model.add(Dense(input_dim=28 * 28, units=689, activation='sigmoid'))
model.add(Dense(units=689, activation='relu'))
model.add(Dense(units=689, activation='relu'))
model.add(Dense(units=689, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=20)
result = model.evaluate(x_test, y_test)
print("\nTest Acc:", result[1])




