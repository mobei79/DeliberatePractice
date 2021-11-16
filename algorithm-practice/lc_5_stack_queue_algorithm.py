# -*- coding: utf-8 -*-
"""
@Time     :2021/11/11 15:40
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
class StactQueue:
    """
    了解java中常见的stack queue容器；
    栈实现队列；队列实现栈；
    栈经典题型：
        系统中括号 花括号 路径 递归等都是 栈实现的。
        括号匹配问题：
        字符串去除相邻重复项；
        逆波兰表达式
    队列经典题型：
        滑动窗口中的最大值？
        前k高频 ： 优先级队列（其实就是堆）




    """


class MyQueue:
    """
    232. 用栈实现队列
    请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

    实现 MyQueue 类：

        void push(int x) 将元素 x 推到队列的末尾
        int pop() 从队列的开头移除并返回元素
        int peek() 返回队列开头的元素
        boolean empty() 如果队列为空，返回 true ；否则，返回 false

    思路：
        python中没有队列，使用数组实现即可。
    """

    def __init__(self):
        """
        in主要负责push，out主要负责pop
        """
        self.stack_in = []
        self.stack_out = []

    def push(self, x: int) -> None:
        """
        有新元素进来，就往in里面push
        """
        self.stack_in.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.empty():
            return None

        if self.stack_out:
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()

from collections import deque
class MyStack:

    def __init__(self):
        """
        Python普通的Queue或SimpleQueue没有类似于peek的功能
        也无法用索引访问，在实现top的时候较为困难。

        用list可以，但是在使用pop(0)的时候时间复杂度为O(n)
        因此这里使用双向队列，我们保证只执行popleft()和append()，因为deque可以用索引访问，可以实现和peek相似的功能

        in - 存所有数据
        out - 仅在pop的时候会用到
        """
        self.queue_in = deque()
        self.queue_out = deque()

    def push(self, x: int) -> None:
        """
        直接append即可
        """
        self.queue_in.append(x)

    def pop(self) -> int:
        """
        1. 首先确认不空
        2. 因为队列的特殊性，FIFO，所以我们只有在pop()的时候才会使用queue_out
        3. 先把queue_in中的所有元素（除了最后一个），依次出列放进queue_out
        4. 交换in和out，此时out里只有一个元素
        5. 把out中的pop出来，即是原队列的最后一个

        tip：这不能像栈实现队列一样，因为另一个queue也是FIFO，如果执行pop()它不能像
        stack一样从另一个pop()，所以干脆in只用来存数据，pop()的时候两个进行交换
        """
        if self.empty():
            return None

        for i in range(len(self.queue_in) - 1):
            self.queue_out.append(self.queue_in.popleft())

        self.queue_in, self.queue_out = self.queue_out, self.queue_in  # 交换in和out，这也是为啥in只用来存
        return self.queue_out.popleft()

    def top(self) -> int:
        """
        1. 首先确认不空
        2. 我们仅有in会存放数据，所以返回第一个即可
        """
        if self.empty():
            return None

        return self.queue_in[-1]

    def empty(self) -> bool:
        """
        因为只有in存了数据，只要判断in是不是有数即可
        """
        return len(self.queue_in) == 0

def peek(self) -> int:
    """
    Get the front element.
    """
    ans = self.pop()
    self.stack_out.append(ans)
    return ans

def empty(self) -> bool:
    """
    只要in或者out有元素，说明队列不为空
    """
    return not (self.stack_in or self.stack_out)
