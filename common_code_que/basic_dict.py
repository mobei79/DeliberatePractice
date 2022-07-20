
# -*- coding: utf-8 -*-
"""
@Time     :2022/3/8 10:25
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import pandas as pd

import numpy as np

df = pd.DataFrame({
    'colA' : list('AABCA'),
    'colB' : ['X',np.nan,'Ya','Xb','Xa'],
    'colC' : [100,50,30,50,20],
    'colD': [90,None,60,80,50]
})
print(df)
print(df.to_dict())
a = df.to_dict()
b = df.to_dict(orient='list')
c = df.to_dict(orient="series")
print(c)
new_df = pd.DataFrame(b)
print(new_df)


class Sameple:
    def __enter__(self):
        print("In __enter__()")
        return "Foo"
    def __exit__(self, type,value, trace):
        print("In__exit__()")
    def get_sample():
        return Sameple()
with Sameple.get_sample() as sample:
    print(sample)