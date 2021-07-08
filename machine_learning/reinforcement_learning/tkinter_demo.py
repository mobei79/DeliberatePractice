# -*- coding: utf-8 -*-
"""
@Time     :2021/7/8 16:38
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

from tkinter import *
def tk_demo():
    root = tk.Tk() # 创建窗口对象的背景色
    li  = ['c','python','php','html','SQL','java']
    movie = ['xixi','haha','lala']
    listb = Listbox(root) #  创建列表组件
    for item in li: # 往组件中插入数据
        listb.insert(0,item)
    listb.pack()# 将小部件放置到主窗口中
    root.mainloop()# 进入消息循环

def foo():
    print("Start ...")
    while True:
        res = yield 7
        res = yield 9
        print("res:", res)
g = foo()
print(g)
print(next(g))
print(next(g))
print(next(g))