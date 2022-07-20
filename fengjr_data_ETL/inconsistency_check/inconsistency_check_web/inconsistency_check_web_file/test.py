# -*- coding: utf-8 -*-
"""
@Time     :2021/5/29 17:54
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 1500)
info = []
# info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19080750010579263","2019-08-07 15:10:23.0","伊春市","黑龙江省伊春市西林区"])
# info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19031150003946155","2019-03-11 14:30:54.0","伊春市","黑龙江省伊春市西林区"])
# info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19112250016532904","2019-11-22 21:33:09.0","伊春市","江苏省徐州市睢宁县"])
# info.append(["+0Y3oMLnSwE9r4MepU0+ZUszgGM6OjZYzVrah/daHp4=","CM19081650011153582","2019-08-16 10:34:56.0","南京市","江苏省徐州市睢宁县"])

# 测试空值
info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19080750010579263","2019-08-07 15:10:23.0",None,"黑龙江省伊春市西林区"])
info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19031150003946155","2019-03-11 14:30:54.0","伊春市","黑龙江省伊春市西林区"])
info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19112250016532904","2019-11-22 21:33:09.0","伊春市",None])
info.append(["+0Y3oMLnSwE9r4MepU0+ZUszgGM6OjZYzVrah/daHp4=","CM19081650011153582","2019-08-16 10:34:56.0","南京市","江苏省徐州市睢宁县"])

df = pd.DataFrame(info, columns=['id', 'income_no', 'income_tm']+["company_city","resident_city"], index=['a','b','c','d'])

# df[:,'income_tm'] = pd.to_datetime(df['income_tm'])

print df
# print df.at['a',"company_city"]
# print df.isnull().any(axis=1)
# print df.notnull().all(axis=0)
print "********************"
# print df.dropna(axis=0,how="any")
# print df.fillna(value="1")
# print df.duplicated()
df = df.fillna("辛集市")
# print df["company_city"]
doc = {"company_city":["辛集市","伊春市","南京市"], "code":[11,22,33]}
code = {"辛集市":11,"伊春市":22,"南京市":33}
df["code"] = df["company_city"].map(code)
def complex(x):
    return x*10
# print type(df.at["a","code"])
def axisx(x):
    if isinstance(x, int) : x = x+1000;
    if isinstance(x, str) : x = x+"xixixi"
    return x
df["code_max"] = df["code"].map(complex)
df.loc['e',:] = df.loc["a",:].map(axisx)
df["apply"] = df["code"].apply(lambda x:x+10000)
print df







# print type(df.columns)
# print type(df.index)
# print type(df.values)
# print df.shape
# print df.size

# print "选择单个元素时，df[列][行]" \
#       "选取行时直接在df后的[]中写一个值或者分号隔开，或者使用布尔数组；" \
#       "选取列时直接在df后的[]写列名，或者列明数组，或者Callable对象" \
#       "选取区域使用iloc loc ix，逗号区分行列，先行后列 "
# print df[:'a']
# print type(df.loc['a',:])
# print type(df.loc[['a','c'],:])
# print type(df[["id","income_tm"]])
# print type(df.loc["a",:])
# print df.iloc[[1,2,3],:]
# print df[[True,True,True,True]]
# print df.loc[:,["id","income_tm"]]
# print type(df.iloc[2,[1,2,3]])
# print df.loc[df["id"]!="++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=",:]

df = pd.DataFrame({"a":[1,1,2,2,3,3],"b":[11,22,33,44,55,66]},index=list("abcdef"))
print df.groupby("a").shift(1)