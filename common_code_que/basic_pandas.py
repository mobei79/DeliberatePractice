# -*- coding: utf-8 -*-
"""
@Time     :2021/12/16 16:10
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# datas = pd.date_range('20130101',periods=5)
# se1 = pd.Series([1, 3, 4, np.nan, 6], index=datas)
# print(se1.unique)



# se11 = pd.Series(np.random.rand(4))
# print(se11)
#
# pd11 = pd.DataFrame(np.random.randn(2, 3), columns=list("ABC"))
# print(pd11)
# pd11 = pd.DataFrame(np.random.randint(1,10,size=(2, 3)), index=["a", "b"], columns=list("ABC"))
# print(pd11)
df = pd.DataFrame(data={"name":["张三","李四","王五","赵六"], "age":[1,np.nan,3,6],
                        "score":[19, 29, 29, 59]}, index=list("abcd"))
df3 = pd.DataFrame(data={"name":["张 三","李 四","王 五","赵 六"], "age":[1,np.nan,3,6],
                        "score":[19, 29, 39, 59]}, index=list("abcd"))
# print(df.describe())
# df.describe()["min": "max"].plot.line()
# print(df.dtypes)
# print(df.groupby(["age","name"]).score.mean())
# df.loc['ee'] = ["hello",2,3]

# df.sort_values(["age",'score'], ascending=[False, False], inplace=True)
# new_df = df.score.astype("int32")
# a = df.name.replace("张三","zhangsansan")
# new_df = df.rename(columns={'name':"Name"}, index={"a":"aa"})

df3[["A","B"]]= df3.name.str.split(" ",n=-1,expand=True)

df3["C"] = df3["A"] + df3["B"]
# print(df)
# print(df.fillna(method='bfill', axis=0, inplace=True))

# df = pd.get_dummies(df, prefix="score")
print(df)
# codes, uniques = pd.factorize(df["score"])
# print(codes, uniques)

# size_map = {19:111, 29:222, 59:333}
# df["score"] = df["score"].map(size_map)

def test(x, **kwargs):
    for n in kwargs:
        x += kwargs[n]
    return x

# new_df = df..apply(test,axis=1, n1=2, n2=100)
# print(new_df)

# a = df.select_dtypes(exclude='object').agg(["mean","min"])
a = df.groupby("name").age.transform(["mean"])
print(df)

# 取行
# print(df["a"]["name"])
# df.iloc[0:2,0:2]  #  先行后列
#             df.iloc[[0,1],[1,3]] # 先行后列，iloc对于隐式索引的操作
#         取区域
#             df.loc["a":'b',"A":"C"]  # loc对于显式索引的操作，df[行区域，列区域] 先列后行
#             df.loc[["a",'b'],["A","C"]]   # 先行后列


# 取列
# print(df["name"])
# print(df[["name","age"]])
# print(df.iloc[:,[1,2]])
# 取区域
# print(df.iloc[1,1])



#
#
# # print(pd33[["name", "age"]])
# # print(pd33['a':'a'])
# # print(pd33["name"]["b"])
#
# print(type(df.loc[:,"name":"age"]))



