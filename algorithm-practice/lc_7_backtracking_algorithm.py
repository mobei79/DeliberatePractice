# -*- coding: utf-8 -*-
"""
@Time     :2021/11/12 10:26
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
class BackTracking:
    """
    知识点：
        回溯时递归的副产物；
        回溯法是暴力搜索，并不高效，最多剪枝一下；
        回溯三部曲：
        back_tracking(参数)：
            if (终止条件)：
                存放结果
                :return
            for (选择本层中的元素)
                处理节点
                back_tracking(路径，选择列表) 递归
                回溯，撤销处理结果
    常用问题：
        组合问题：N个数里面按照一定规律找出k个数的集合
        排列问题：N个数里面按照一定规则全排序，有几种排序方式
        切割问题：字符串按规律有多少种切割方法
        子集问题：N个数里有多少符合规则的子集
        期盼问题：数独

    """
