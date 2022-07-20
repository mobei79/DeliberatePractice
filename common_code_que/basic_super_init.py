# -*- coding: utf-8 -*-
"""
@Time     :2022/2/24 14:47
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""


"""
在写神经网络代码时，经常看到如下代码：
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass
要看懂这三行代码需要了解三个东西：
    self参数
    __init__ ()方法
    super(Net, self).init()
self参数：
    指的是实例Instance自身，在python语法中，函数的第一个参数是实例对象本身，约定俗成写作self；即类中方法的第一个参数一定是self，不能省略。
    * 代表实例本身，不是类；可以用this代替，但是不建议这样写；

__init__()方法
    创建类之后，通常会创建一个 __init__()方法，这个方法会在创建类的实例时 自动执行；init方法必包含self；
    如果在__init__方法中传入参数如name，创建这个类实例的时候也需要传入这个参数。
    哪些操作在创建实例的时候就要进行，就写在__init__()方法中。eg.神经网络的参数

super(MyClass, self).__init__()
    指首先找到MyClass的父类FClass，然后把类MyClass的实例对象self转换为父类FClass的类对象，然后”被转换“的类FClass对象调用自己的__init__()函数。
    简单说：子类把父类的__init__()放到自己的__init()__中，这样子类就有了父类init中的东西。

回到上面的例子：
    自定义的Net类继承了nn.Model，super(Net, self).__init__()就是对继承自父类nn.Model的属性进行初始化，
    而且使用nn.Model的初始化方法来初始化继承的属性。
    子类继承了父类所有的属性和方法，父类属性自然会用父类方法来进行初始化。
"""
"""
常见写法：
class A(Base):
    Base.__init__(self) # 写法1
    super(A, self).__init__() # 写法2
写法2：继承父类所有的特性(而不是基类)，并且避免重复继承。

"""
class Base(object):
    def __init__(self):
        print("Base create")


class childA(Base):
    def __init__(self):
        print("init A")
        super(childA, self).__init__()
        print("init a end")

class childB(Base):
    def __init__(self):
        print("init B")
        Base.__init__(self)
        # super(childB, self).__init__()
        print("init B end")

ca = childA()
cb = childB()