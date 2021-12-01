# -*- coding: utf-8 -*-
"""
@Time     :2021/10/28 16:18
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
class parent:
    def __init__(self):
        print("im forther")
        self.name = 'parent'

    def getName(self):
        print(self.name)

    class child:
        def __init__(self):
            print("im son")
            self.name = 'child'

        def getName(self):
            print(self.name)


if __name__ == '__main__':
    # child =  parent.child()
    # child.getName()

    p = parent()
    # p.getName()
    c = p.child()
    c.getName()