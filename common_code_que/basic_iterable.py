# -*- coding: utf-8 -*-
"""
@Time     :2021/9/6 23:28
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

""" 可迭代对象、迭代器、生成器
1. 可迭代对象：
    可迭代取值的数据类型：字符串、数组、元组、字典，range(100)；以及文件（迭代器）；
        str = "string" # 可迭代对象
        iter = str.__iter__()   #调用__iter__()生成迭代器
        iter.__next__()         #调用__next__()取一个值，这样可以不依赖索引取值
    超过范围之后，会报错StopIteration；
    可迭代对象iterable都内置了__iter__()方法。【迭代器也内置了,iter is iter.__iter__()】
    也可以使用iter()和next()，他的底层是上面的实现；缺点是迭代器只能取一次；
        iter = iter(str)
        a1 = next(iter)
    for循环，就是迭代器循环，其底层实现原理：1.先调用in后面对象的__iter__()方法生成迭代器；2.调用__next__()将值赋给变量名；3.for循环会自动捕获懿诚结束；
    因为迭代器的__iter__()是他自己，所以迭代器和可迭代对象都可以通过for循环；
    *** 可迭代对象只有__iter__()方法，没有__next__方法；除文件外其他的容器都是可迭代对象；
2. 迭代器优点：不依赖索引，在内存中只占用一个空间；
    迭代器用完之后就没了，比如文件就是迭代器；可迭代对象调用完之后，在生成迭代器就行；
    *** 迭代器有__iter__()方法和__next__方法；迭代器对象一定是可迭代对象，可迭代的对象（只需有iter）不一定是迭代器对象；
3. 生成器
    生成器就是一种自定义的迭代器，函数中有yield关键字，调用函数不会执行函数体代码，会得到一个返回值，该返回值就是生成器对象；内容就是yield后面的值，通过迭代访问；
    next会触发函数的执行，知道碰到yield停下来，将返回后面的值当做本次next的值
    总结：
        yield只能在函数中使用；
        yield提供一种自定义迭代器的解决方案；可以保存函数的暂停的状态；
        yield对比return：
            相同：都可以方绘制，类型和数量都没有限制
            不同：yield可以返回多次，return只能返回一次函数就结束了。
"""
def fun(n):
    loop, a, b = 0, 1, 1
    while loop < n:
        yield a
        a, b = b , a+b
        loop+=1

my_yield = fun(10)
print(my_yield)

print(list(my_yield))
# 迭代器取完了。后续for循环没值
for i in my_yield:
    print(i)
