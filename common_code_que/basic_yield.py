# -*- coding: utf-8 -*-
"""
@Time     :2021/9/2 19:58
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
'''
使用yield的函数是一个生成器generator，生成器返回的是一个迭代器（函数返回值不是某个特定值，而是一个迭代器，只能迭代操作）
    原理：再调用生成器的过程中，每次遇到yield时，函数就会暂停并保存当前所运行的信息，
    挂起函数状态，返回yield值（也就是一个迭代器），并等待下一次执行next()方法时，再继续进行；
'''

def yield_fun():
    print("yield function start")
    yield 1
    print("yield function mid")
    yield 2
    print("yield function end")

# 首先实例这个函数，每次访问才会放回对应的yield的值。返回yield值之后，后面的函数就不执行啦
f = yield_fun()
rst_1 = next(f)
print(rst_1)
rst_2 = next(f)
print(rst_2)

''' out
yield function start
1
yield function mid
2

Process finished with exit code 0
'''