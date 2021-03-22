# -*- coding: utf-8 -*-
"""
@Time     :2021/2/19 17:31
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

"""
python 字典defaultdict

data = [("p", 1), ("p", 2), ("p", 3),
        ("h", 1), ("h", 2), ("h", 3)]
转化为
result = {'p': [1, 2, 3], 'h': [1, 2, 3]}
"""
# 方法1
result = {}
data = [("p", 1), ("p", 2), ("p", 3),
        ("h", 1), ("h", 2), ("h", 3)]
for (k, v) in data:
    print(result.setdefault(k, []))
    print(result.get(k, default=[]))
    result.setdefault(k, []).append(v)
# print(result)
# 方法2
from collections import defaultdict
result = defaultdict(list)
for (k, v) in data:
    result[k].append(v)

print("*************")
for k, v in result.items():
    print(k, v)


