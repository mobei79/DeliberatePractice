# -*- coding: utf-8 -*-
"""
@Time     :2021/9/6 10:07
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
# 闭包函数 closure function：是函数式编程中重要的语法结构。
'''
作用：内部函数包含外部作用域而非全局作用域名字的引用。
判断：__closure__:是一个cell对象，表示是闭包函数。返回None表示不是。
1. 必须要有内嵌函数；
2. 内嵌函数需要引用该嵌套函数上一级namespace中的变量（这个变量可以是函数）；
3. 闭包函数必须返回内嵌函数；
'''
def closure_out(fun):
    print("闭包函数-外层-start")
    def closure_inner():
        print("闭包函数-内层-start")
        # fun()
        print(fun)
        print("闭包函数-内层-end")
    print(closure_inner.__closure__)
    print("闭包函数-外层-end")
    return closure_inner

# def fun():
#     print("内层函数所调用的局部作用域内的参数")
fun = "内层函数所调用的局部作用域内的参数"
rst_fun = closure_out(fun)
print(rst_fun)
rst_fun()

# 装饰器 就是一个特殊的闭包函数
'''
什么是？
    其实就是一个函数，只是这个函数不是自己使用，而是给其他函数”添加功能“的
    器：指的是工具，程序中函数就是具备某功能的工具
    装饰：指的是为被装饰器对象添加额外的功能；
为什么需要装饰器：
    软件的维护应该遵循“开放封闭原则”
        软件一旦上线后，对修改源代码是封闭的，对于扩展功能是开放的；
        这就用到了装饰器
    装饰器的实现遵循两大原则：
        1. 不修改别装饰对象的源代码
        2. 不修改别装饰对象的调用方式
    装饰器在遵循1,2原则的基础上为被装饰对象添加新功能；
'''
def jianshen(): # 增加了jianshen功能，但是改变了原调用方式；
    print("jianshen")
    run()

def run():
    print("jianshen") # 如果在run函数中新增加jianshen功能，就改变了原代码
    print("run")

from datetime import datetime
def run_time(func):
    def new_fun():
        start = datetime.now()
        print("开始时间%s", start)
        func()
        end = datetime.now()
        print("结束时间%s", end)
        total_time = end - start
        print("共耗时%s",total_time)
    return new_fun

@run_time #装饰谁就@谁； 其底层逻辑my_type = run_time(my_type)
def my_type():
    print("------type-------")
    for i in range(10000):
        type('hello')
@run_time
def my_instance():
    print("------isinstance-------")
    for i in range(10000):
        isinstance('hello', str)
'''
1. 没有改变my_type中的代码内容;
2. 命名为my_type也没有改变其调用方式；
“my_type = run_time(my_type)”
这是调用装饰器的底层逻辑
简便方法，我们@run_time就行
'''

my_type()
my_instance()
my_type = run_time(my_type)
my_type()