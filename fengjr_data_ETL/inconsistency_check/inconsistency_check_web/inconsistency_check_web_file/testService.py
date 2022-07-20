# -*- coding: utf-8 -*-
"""
@Time     :2021/5/30 13:30
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
prop = "resident_city"
id = "++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88="
ids = [id]
props = [prop]

info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19080750010579263","2019-08-07 15:10:23.0","黑龙江省伊春市西林区1"])
info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19031150003946155","2019-03-11 14:30:54.0","黑龙江省伊春市西林区2"])
info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19031150003946156","2019-03-11 14:30:54.0","黑龙江省伊春市西林区3"])
info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19031150003946157","2019-03-11 14:30:54.0","黑龙江省伊春市西林区2"])
# info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19031150003946155","2019-03-11 14:30:54.0","黑龙江省伊春市西林区2"])
# info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19112250016532904","2019-11-22 21:33:09.0","江苏省徐州市睢宁县"])
# info.append(["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=","CM19112250016532904","2019-11-22 21:33:09.0","江苏省徐州市睢宁县"])
df = pd.DataFrame(info, columns=['id', 'income_no', 'income_tm']+["resident_city"])
df.loc[:, 'income_tm'] = pd.to_datetime(df['income_tm'])

nn = df[df[prop].notnull() & (df[prop].str.len() > 0)]
income_cnt = nn.groupby(["id"])["income_no"].count()
ge2_id = income_cnt[income_cnt>1].index.values # 筛选出进件大于1次的id
ge2_df = nn[nn['id'].isin(ge2_id)].sort_values("income_tm").copy()
prop_shift = prop + "_shift"
shift = ge2_df.groupby("id")[prop].shift(1)
# ge2_df.loc[:,prop_shift] =
ge2_df.loc[:, prop_shift] = shift
def equal(s1, s2, pct):
    return s1 != s2
print "*****"
print equal("12","12",1)
# print pd.notnull(ge2_df[prop_shift])
# print equal(ge2_df[prop],ge2_df[prop_shift],1)
# print ge2_df['income_tm']
# print ge2_df['income_tm'].shift(1)
# print pd.notnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(1))
# print (ge2_df['income_tm'] - ge2_df['income_tm'].shift(1) < pd.to_timedelta(1000, 'D'))

diff_prop = max(pd.notnull(ge2_df[prop_shift])
                & equal(ge2_df[prop],ge2_df[prop_shift],1)
                & pd.notnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(1))
                & (ge2_df['income_tm'] - ge2_df['income_tm'].shift(1) < pd.to_timedelta(1000, 'D'))
)
# print diff_prop
# rst = {ge2_df["id"].unique()[0]: 1 if diff_prop else 0}
# print ge2_df['income_tm'] - ge2_df['income_tm'].shift(1)
#
# print pd.notnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(1))
# print pd.notnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(0))
#
# print max(pd.notnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(1)) & pd.notnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(0)))
# print {ge2_df['id'].unique()[0]: 1 if diff_prop else 0}
#
# print "***********************"
# print pd.notnull(ge2_df[prop_shift])
# print  equal(ge2_df[prop],ge2_df[prop_shift],1)
# print ge2_df['income_tm'] - ge2_df['income_tm'].shift(1)
# print (ge2_df['income_tm'] - ge2_df['income_tm'].shift(1) < pd.to_timedelta(1000, 'D'))
# print type(pd.notnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(1)))
# a = pd.notnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(1))
# b = pd.isnull(ge2_df['income_tm'] - ge2_df['income_tm'].shift(1))
# print a&b
# rst = {ge2_df['id'].unique()[0]: 1 if diff_prop else 0}
# if len(df['id'].unique())>1:
#     rst.update({i: 0 for i in df['id'].unique() if i != ge2_df.loc[0, 'id']})
# result = {'data': rst, 'rst': 1}
# # logger.info('query result: {} only 1 id has >1 income'.format(result))
# print result

# nedf = pd.DataFrame({'s1':["aa",'bb','aa'],'s2':[None,"aa",'bb']})
# # nedf.fillna('',inplace=True)
# print nedf
# print nedf.shift(1)
