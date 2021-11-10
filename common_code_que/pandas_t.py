# -*- coding: utf-8 -*-
"""
@Time     :2021/10/21 11:05
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

import pandas as pd
import numpy as np

"""
series
创建：三种方式list、np、dict
索引：
    索引无非就是显式和隐式之分：
    隐式：
        又分为：取单个值，取一组值{直接使用数组，区间数组，使用iloc数组}，
        se[0]
        se[[0,2,3]]
        se[2:4] #左闭右开
        se.iloc[1:3]
        se.iloc[[2,3,4]]
    显示：
        se["A"]
        se[["A", "B"]]
        se["A":"C"]
        se.loc["A":"D"]
        se.loc[["A", "B"]]      # loc返回的是series
    基本操作
        se.head(n)
        se.tail(n)
        se.unique() 返回去重数组
    空值处理
        se.notnull()    返回的是 数组
        se.isnull()
        se[] 
    注：取值都是方括号；方法类的都是圆弧括号；单值或者使用区间直接在方括号中输入；
"""
se1 = pd.Series([1, 3, 4, np.nan, 6], index=["A", "B", "C", "D", "E"])
se2 = pd.Series(np.random.randint(1,10,size=(3,)), index=["A", "b", "C"])
se3 = pd.Series({"a":1, "b":2})



# print(se1[se1.notnull()])

"""
dataframe
创建：
    使用ndarray、字典、数组
切片和索引
    隐式索引：
        分为取单值、取行、取列、取区域
        
        df[0:3]        # 取行
        df.iloc[0:3]        # 取行
        
        # df[[0,1]]       # 取列 series中可以使用这种方法取区间
        df.iloc[[0,2],[0,2]] 或者 df.iloc[0:2,0:2]
    
        分号区间:可以直接写，但是数组需要在括起来。  
    取单值：
        df["A"]["b"] 显式取单值
        df.iloc[1,2]  隐式取单值
    取行：
        隐式：
            df[0:3]
            df.iloc[0:3]
        显式：
            df[:"b"]
            df["a":"b"]
        布尔数组：
            df[[each > 20 for each in df["A"]]
            df[(df["A"] > 20) & df["isM"] == "no"]
    取列：
        显式：
          df["name"]   
          df[["name","age"]]   
        callable
            df[lambda df : df.columns[0]]  选取第一列   
    取区域：
        df.iloc[:3,:]   df.iloc[[1,2,3],:]
        df.loc[]
        df.ix[]
    
        
"""
df1 = pd.DataFrame(data=np.random.randint(1,10,size=(4,5)),
                   index=["a", "b", "c", "d"],
                   columns=["A","B","C","D","E"],dtype='int64' #dtype / copy=Flase
                   )
df2 = pd.DataFrame({"a":[1,2,3], "b":[4,5,6], "c":[7,8,9]}, index=["A","B","C"])
print(df1)
print("*"*20)
# print(df1.iloc[0,2])
# print(df1[:"a"])
# cell
print(df1.iloc[1,2])
print(df1["C"]["b"])
# row
print(df1[:2])
print(df1.iloc[1:2])
print(df1[:"c"])
print(df1[[True, False, False, True]])
# col
print(df1["A"])
print(df1[["A", "B", "C"]])
print("$$"*20)
# area
print(df1.iloc[1:3,2:4])
print(df1.loc[["a","b"],["C", "D", "E"]])
