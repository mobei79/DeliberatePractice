# -*- coding: utf-8 -*-
"""
@Time     :2021/2/26 10:54
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
class Solution:
    def transpose(self, matrix : list[list[int]]) -> list[list[int]]:
        M, N = len(matrix), len(matrix[0])
        res = [[0]*M for i in range(N)]
        for i in range(M):
            for j in range(N):
                res[j][i] = matrix[i][j]
        return res
    """
    使用numpy的 transpose函数
    """
    def transpose_pro(self, matrix: list[list[int]]) -> list[list[int]]:
        import numpy as np
        return np.transpose(matrix).tolist()

class CQueue():
    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def appendTail(self, value: int) -> None:
        self.stack_in.append(value)
    def deleteHead(self) -> int:
