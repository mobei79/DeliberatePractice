
# -*- coding: utf-8 -*-
"""
@Time     :2021/9/23 10:58
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import pandas as pd
import numpy as np

def dict2order():
    """
    sorted() 函数对所有可迭代的对象进行排序操作。
    sort 与 sorted 区别：
        sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
        list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
    sorted 语法：
        sorted(iterable, key=None, reverse=False)
        key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
        reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
    https://www.runoob.com/python3/python-sort-dictionaries-by-key-or-value.html
    :return:
    """
    # key_value = {'a':1,'b':5,'c':3}
    # 声明字典
    key_value = {}
    # 初始化
    key_value[2] = 56
    key_value[1] = 2
    key_value[5] = 12
    key_value[4] = 24
    key_value[6] = 18
    key_value[3] = 323
    print("按照key值进行排序：")
    for i in sorted(key_value): # 默认返回传入数据的key，如果是dict.items() 则返回一对数据；
        print(i, key_value[i], end="\n")


    print("按照value键进行排序：")
    for i in sorted(key_value.items(),key=lambda d:d[1], reverse=False):
        print(i[0], i[1], end="\n")
    # print(key_value.items(), key = lambda kv:(kv[1], kv[0]))
    for item in sorted(key_value.items(), key= lambda kv:(kv[1], kv[0])):
        print(item[0], item[1], end="\n")


    print("字典列表排序")
    lis = [{"name": "Taobao", "age": 100},
           {"name": "Runoob", "age": 7},
           {"name": "Google", "age": 100},
           {"name": "Wiki", "age": 200}]
    # 通过age 升序排列
    print(sorted(lis, key= lambda i : i['age']))
    print("\r")
    # 通过name和age
    print(sorted(lis, key = lambda i:(i["name"],i["age"])))

    # list 倒叙排列
    example_list = [5, 0, 6, 1, 2, 7, 3, 4]
    result_list = sorted(example_list, key=lambda k:k*-1)
    print(result_list)


def NP_zhenghe():
    pass

def PD_NAN_rate():
    """
    统计特征缺失率
    1. 如何表示缺失率
    python使用numpy和pandas处理数据，python中None表示确实值，使用None创建列表，无法调用mean、sum、min、max等数值计算函数，报错TypeError；
    NumPy提供了np.nan表示缺失值，但是本质上他是浮点值；包含np.nan的数组进行数值运算不会引发类型异常；但是涉及np.nan的运算结果还是np.nan,需要忽略掉才行如下：
        arr = np.array([1, 2, np.nan, 4])
        # 结果是np.nan
        print(np.mean(arr), np.sum(arr), np.min(arr), np.max(arr))
        # 计算时剔除np.nan，获得正确结果
        print(np.nanmean(arr), np.nansum(arr), np.nanmin(arr), np.nanmax(arr))
    Pandas 中可以使用None和np.nan表示缺失值，如果数组中有None会自动转换为np.nan
    2. 处理缺失值
        由上可知，pandas中使用None和np.nan表示缺失值，pd.Series pd.DataFrame 均提供了几种常见的方法 侦测 剔除 填充缺失值
        isnull()    返回布尔数组，返回True False
        notnull()   筛选非缺失，isnull() 逆运算
        dropna()    # 剔除缺失值， 样本空间足够大才建议删除
        fillna()
        2.1 剔除缺失值：
    :return:
    """

    # path = "D:\Program Files\pythonwork\DeliberatePractice\\fengjr_data_ETL\overdue_predict\data\\raw_feature_2020-12-02_2020-12-07.csv"
    # df = pd.read_csv(path)
    ser = pd.Series([1, None, 3, np.NAN, pd.NaT, pd.NA, 7])
    df = pd.DataFrame({'a':pd.Series([np.NAN, 2, pd.NaT, '', None ,'I stay']),
                       'b':pd.Series([np.NAN, 2, 3, 4, None, 79])})
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df)

    print("统计列缺失值")
    print(df.notnull())
    print(df.isnull().sum())        # 按列统计缺失值
    print(df.isnull().sum(axis=0))  # 按列统计缺失值
    print(df.isnull().any())
    print(df["a"].isnull().sum())   # 统计某一列的缺失值

    print("统计行缺失值：")
    print(df.isnull().sum(axis=1)) # 每一行有多少个缺失值的值，即按行统计缺失值

    print("统计整个df的缺失值：")
    print(df.isnull().sum().sum())

    print("统计缺失值 使用count")
    print(df.count(axis=1)) # 计算data每一行有多少个非空的值，即按行统计非空值
    print(df.count())       # 按列统计非空值
    print(df.shape[1] - df.count())

    # 删除缺失值
    print("\r")
    df_row = df.dropna() # 剔除任意包含缺失值的行
    df_col = df.dropna(axis=1) # 剔除包含缺失值的列
    df_thresh = df.dropna(thresh=1) # 当非缺失值的数量大于等于thresh，保留该行
    print(df_row, df_col, df_thresh)

    # 填充缺失值
    df_0 = df.fillna(0) # 填充 0
    df_ffill = df.fillna(method="ffill") # 向前填充forward
    df_bfill = df.fillna(method="bfill") # 向后填充backward fill
    df_mean = df.fillna(df.mean) # 计算每一列的均值，用均值填充




if __name__ == "__main__":
    # dict2order()

    PD_NAN_rate()