# -*- coding: utf-8 -*-
"""
@Time     :2021/12/15 11:33
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import sys
import traceback
import logging
import threading

def func1():
    raise Exception("--- func1 exception ---")
    # raise NameError("--- func1 exception ---")

def func2():
    func1()

def main1():
    try:
        func1()
    except Exception as e:
        print("普通的打印异常只会显示value值：")
        print(e)

        print("通过sys.exc_info() 函数获取traceback object")
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        print("exc_type : {}".format(exc_type))
        print("exc_value : %s" % exc_value)
        print("exc_traceback_obj : %s" % exc_traceback_obj)

def main2():
    try:
        func2()
    except Exception as e:
        print("通过traceback module打印traceback object 相关信息")
        exc_type, exc_value, exc_tracebace_obj = sys.exc_info()
        # traceback.print_tb(exc_tracebace_obj)

        print("2：借助print_tb方法，打印更加详细的异常信息")
        """
        traceback.print_tb(tb[, limit[, file]])
            tb: 这个就是traceback object, 是我们通过sys.exc_info获取到的
            limit: 这个是限制stack trace层级的，如果不设或者为None，就会打印所有层级的stack trace
            file: 这个是设置打印的输出流的，可以为文件，也可以是stdout之类的file-like object。如果不设或为None，则输出到sys.stderr。
            作者：geekpy
            链接：https://www.jianshu.com/p/a8cb5375171a
            来源：简书
            著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
        """
        # traceback.print_tb(tb=exc_tracebace_obj)

        print("3：借助print_exception方法，打印更加详细的异常信息")
        traceback.print_exception(etype=exc_type, value=exc_value, tb=exc_tracebace_obj, limit=2,file=sys.stdout)

        print("4：使用 print_exc() ,他只有两个参数")
        traceback.print_exc(limit=2, file=sys.stdout)

        print("5：使用 format_exc() 来获取一个字符串，保存在log中，")
        logging.error(traceback.format_exc(limit=2))

def my_func():
    raise BaseException("thread exception")


class ExceptionThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        """
        Redirect exceptions of thread to an exception handler.
        """
        threading.Thread.__init__(self, group, target, name, args, kwargs)

        if kwargs is None:
            kwargs = {}
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self._exc = None

    def run(self):
        try:
            if self._target:
                self._target()
        except BaseException as e:
            import sys
            self._exc = sys.exc_info()
        finally:
            #Avoid a refcycle if the thread is running a function with
            #an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def join(self):
        threading.Thread.join(self)
        if self._exc:
            msg = "Thread '%s' threw an exception: %s" % (self.getName(), self._exc[1])
            new_exc = Exception(msg)
            raise(new_exc.__class__, new_exc, self._exc[2])

if __name__ == "__main__":
    # main1()
    # main2()
    t = ExceptionThread(target=my_func, name='my_thread')
    t.start()
    try:
        t.join()
    except:
        traceback.print_exc()

"""
普通的打印异常只能打印少量信息（异常的value值）很难确定那块代码出的问题。
如何更加详细的打印异常信息？？？
    sys.exc_info 和 traceback object
python程序中的traceback信息均来自于 traceback object, traceback object 通常通过函数sys.exc_info()获取。
sys.exc_info() 获取当前处理的exception的相关信息，并返回一个 【元组】，元组数据分别为：异常类型、异常的value值、traceback object；
有了traceback就可以通过traceback module打印格式化的traceback 相关信息。

print("借助print_tb方法，打印更加详细的异常信息")
print("借助print_exception方法，打印更加详细的异常信息")
print_exc 是最常用的，直接自动执行exc_info() 来获得三个参数，最简单。
print("4：使用 print_exc() ,他只有两个参数")
print("5：使用 format_exc() 来获取一个字符串，保存在log中，")
"""

"""
我在Thread.__init__()中尝试忽略Verbose，然后它就可以很好地工作到我的最后。找到我的演示代码和输出。
之前的Python<3在Thread中使用了Verbose，但在3之后。忽略x+ Verbose。
"""