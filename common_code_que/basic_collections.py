# -*- coding: utf-8 -*-
"""
@Time     :2021/11/24 15:02
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from collections import Counter

str = "abcbcaccbbad"
li = ["a","b","c","a","b","b"]
d = {"1":3, "3":2, "17":2}

#Counter获取各元素的个数，返回字典
print ("Counter(s):", Counter(str))
print ("Counter(li):", Counter(li))
print ("Counter(d):", Counter(d))
"""
>>>>>>
Counter(s): Counter({'b': 4, 'c': 4, 'a': 3, 'd': 1})
Counter(li): Counter({'b': 3, 'a': 2, 'c': 1})
Counter(d): Counter({'1': 3, '3': 2, '17': 2})
"""
#most_common(int)按照元素出现的次数进行从高到低的排序，返回前int个元素的字典
d1 = Counter(str)
print("d1.most_common(2):",d1.most_common(2))

#elements返回经过计算器Counter后的元素，返回的是一个迭代器
print ("sorted(d1.elements()):", sorted(d1.elements()))
print ('''("".join(d1.elements())):''',"".join(d1.elements()))

#若是字典的话返回value个key
d2 = Counter(d)
print("若是字典的话返回value个key:", sorted(d2.elements()))

#update和set集合的update一样，对集合进行并集更新
print ("d1.update(sas1):",d1.update("sas1"))