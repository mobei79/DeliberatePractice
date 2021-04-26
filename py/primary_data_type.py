#!/usr/bin/python3

# coding=utf-8
"""
Python2 中默认的编码格式是 ASCII 格式，在没修改编码格式时无法正确打印汉字，所以在读取中文时会报错。
Python3.X 源码文件默认使用utf-8编码，所以可以正常解析中文，无需指定 UTF-8 编码。
"""
import random

""" 一些基础语法
多行语句 Python 通常是一行写完一条语句，但如果语句很长，我们可以使用反斜杠(\)来实现多行语句，例如：
total = item_one + \
        item_two + \
        item_three
在 [], {}, 或 () 中的多行语句，不需要使用反斜杠(\)

使用type(val) 查看数据类型  ；type()不会认为子类是一种父类类型。
使用isinstance(val, int) 判断数据类型   ；isinstance()会认为子类是一种父类类型。

标准数据类型
    Python3 中有六个标准的数据类型：
        Number（数字）
        String（字符串）
        List（列表）
        Tuple（元组）
        Set（集合）
        Dictionary（字典）
    Python3 的六个标准数据类型中：
        不可变数据（3 个）：Number（数字）、String（字符串）、Tuple（元组）；
        可变数据（3 个）：List（列表）、Dictionary（字典）、Set（集合）。

"""

"""
数字类型（Number）
    python 中有四种数字类型：整数、布尔型、浮点数、复数
        ** int (整数), 如 1, 只有一种整数类型 int，表示为长整型；没有 python2 中的 Long；
            python内部对整数的处理分为普通整数和长整数，普通整数长度为机器位长，通常32位；超过这个长度就自动当做长整型处理，长度范围没有限制 \
            整数运算要求精准；浮点数运算可能会四舍五入；
        ** bool (布尔), 如 True；  在 Python2 中是没有布尔型的，它用数字 0 表示 False，用 1 表示 True。到 Python3 中，把 True 和 False 定义成关键字了，但它们的值还是 1 和 0，它们可以和数字相加。
        float (浮点数), 如 1.23、3E-2
        complex (复数), 如 1 + 2j、 1.1 + 2.2j
    数字数据类型是不允许改变的，这意味着如果修改数字数据类型的值，将重新分配内存空间；
    可以使用del val 删除对对象的引用；
    python中变量不需要声明；每个变量在使用前必须赋值，赋值后变量才会被创建；
    
    数字函数
        随机数函数 
            choice(seq)从序列的元素中随机选一个整数； random.choice(rang(10))
            randrange()
"""


num_1 = num_1_copy = random.choice(range(10))
num_2 = random.randrange(1,11,2)
num_3 = random.random()
num_4 = random.shuffle([1, 4, 3, 5, 6])
num_5 = random.uniform(7,9)
# input("\n\n按下enter键继续执行")
print(num_1, end="**")
print(num_1_copy, end="\n")
print("num_2=",num_2, num_3, num_4, num_5)
print(num_4)

"""
字符串 string
    python不支持单字符类型，但是单字符以字符串形式存在；
        不支持对字符串进行改变；
    Python中的字符串用单引号 ' 或双引号 " 括起来；
        使用反斜杠 \ 转义特殊字符；（如果你不想让反斜杠发生转义，可以在字符串前面添加一个 r，表示原始字符串 print(r'Ru\noob')）
        加号 + 是字符串的连接符； 
        星号 * 表示复制当前字符串，与之结合的数字为复制的次数；
"""
print(r'Ru\noob')

"""

"""
