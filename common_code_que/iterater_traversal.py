# -*- coding: utf-8 -*-
"""
@Time     :2021/11/17 17:56
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
"""
不可变数据类型指向值，值不变，则对象不变；值变则对象变；
可变数据类型指向名字，名不变则对象不变；名变则对象变；
可变数据类型 list dict 不可变数据类型 string number tuple
当进行修改时，
    可变数据类型传递的是内存地址，也就是说直接修改内存中的值，并没有开辟新的内存；                                 【在不同的递归循环中，处理的是同一份数据】
    不可变数据类型，并没有改变原内存中的地址，而是开辟一块新的内存，将原址中的值赋值过去，对这块新地址空间中的值进行操作。【也就是在不同的递归循环中，处理的都是传入的备份】
自定义的对象按照不可变数据类型进行处理的（比如定义的数根节点root）；
"""
def test_iterator(cur):
    def traversal(cur, path, result):

        if path[0] == 3:
            return
        else:
            path[0] += 1
        if True:
            traversal(cur, path, result)
            print(path)
        if True:
            traversal(cur, path, result)
            print(path)

    path = [0]
    result = []
    traversal(cur, path, result)
    return result

def test_iterator_back_tracking(cur):
    def traversal(cur, path, result):
        path +=1
        if path==3:
            return
        if True:
            traversal(cur, path, result)
            print(path)
        if True:
            traversal(cur, path, result)
            print(path)

    path = 0
    result = []
    traversal(cur, path, result)
    return result

cur = [7,9,5,0,3,6,0,79,2,3,5,0,12,31,56,37]
print(cur)

print(test_iterator_back_tracking(cur))
# print(test_iterator(cur))