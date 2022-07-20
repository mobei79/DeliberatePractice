# coding:utf-8 -*- 
# @Time : 2020-02-11 16:50

"""处理kg中的数据 """
# import difflib, Levenshtein, numpy as np, pandas as pd
import difflib, numpy as np, pandas as pd

from inconsistency_check_web_file.database import query
from utils.log import logger
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 1500)

def to_df(s1, s2):
    df = pd.DataFrame({'s1': s1, 's2': s2})
    df.fillna('', inplace=True)
    return df


def equal(s1, s2, pct):
    return s1 != s2


def longest_sim(s1, s2, pct):
    df = to_df(s1, s2)
    return df.apply(
        lambda x:
        round(difflib.SequenceMatcher(
            lambda x: x == " ", x.loc['s1'], x.loc['s2'])
              .quick_ratio(), 3), axis=1) < pct


def edit_sim(s1, s2, pct):
    df = to_df(s1, s2)
    return df.apply(lambda x: round(Levenshtein.ratio(x.loc['s1'], x.loc['s2']), 3), axis=1) < pct


def jw_sim(s1, s2, pct):
    df = to_df(s1, s2)
    return df.apply(lambda x: round(Levenshtein.jaro_winkler(x.loc['s1'], x.loc['s2']), 3), axis=1) < pct


def jaccard_sim(s1, s2, pct):
    df = to_df(s1, s2)
    def jaccard(str1, str2):
        str1 = set(str1)
        str2 = set(str2)
        return len(str1.intersection(str2)) / len(str1.union(str2))
    return df.apply(lambda x: round(jaccard(x.loc['s1'], x.loc['s2']), 3), axis=1) < pct


fn_dict = {
    'equal': equal,
    # 'edit': edit_sim,
    'longest': longest_sim,
    'jaccard': jaccard_sim
    # 'jw': jw_sim
}

sim_prop = [
    'company_name',
    'resident_address',
    'company_address',
]

supported_prop = [
    'credit_card_number',
    'company_phone',
    'famliy_relationship',
    'resident_address',
    'highest_eduction',
    'company_address',
    'workmate_phone',
    'workmate_name',
    'family_name',
    'work_relationship',
    'family_phone',
    'profession_station',
    'company_name',
    'business_source',
    'is_married',
    'company_town',
    'start_work_tm',
    'channel_code',
    'loan_use',
    'job_salary',
    'company_city',
    'resident_town',
    'income_address',
    'device_address',
    'company_province',
    'mobile_prov',
    'mobile_encrypt',
    'mobile_city',
    'company_name',
    'resident_address',
    'company_address',
    'company_name',
    'resident_address',
    'company_address'
]

def query_diff(id, prop, days, pct=0.9, sim_fn='equal'):
    """单属性字符串查询"""
    logger.info('query params: {}'.format(id, prop, days, pct, sim_fn))
    if prop not in supported_prop:
        result = {'data': None, 'rst': 0}
        logger.info('query result: {}, not supported prop'.format(result))
        return result

    ids, props = id, prop
    if isinstance(id, str): ids = [id]
    if isinstance(prop, str): props = [prop]
    sim_fn = fn_dict.get(sim_fn, None)
    assert sim_fn is not None, 'similarity function {sim_fn} not supported'

    # df = query(ids, props)
    info = []
    info.append(
        ["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=", "CM19080750010579263", "2019-08-07 15:10:23.0", "黑龙江省伊春市西林区"])
    info.append(
        ["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=", "CM19031150003946155", "2019-03-11 14:30:54.0", "黑龙江省伊春市西林区"])
    info.append(
        ["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=", "CM19112250016532904", "2019-11-22 21:33:09.0", "江苏省徐州市睢宁县"])
    df = pd.DataFrame(info, columns=['id', 'income_no', 'income_tm'] + ["resident_city"])
    df.loc[:, 'income_tm'] = pd.to_datetime(df['income_tm'])
    nn = df[df[prop].notnull() & (df[prop].str.len() > 0)]
    income_cnt = nn.groupby(['id']).count()['income_no']
    ge2_id = income_cnt[income_cnt > 1].index.values
    if len(ge2_id) == 0:  # 均进件一次
        result = {'data': dict(zip(df['id'].unique(), [0]*len(df['id'].unique()))), 'rst': 1}
        logger.info('query result: {}, all income only once'.format(result))
        return result

    ge2_df = nn[nn['id'].isin(ge2_id)].sort_values('income_tm').copy()
    shift = ge2_df.groupby('id')[prop].shift(1)
    prop_shift = prop + '_shift'
    ge2_df.loc[:, prop_shift] = shift
    if len(ge2_df['id'].unique()) == 1:  # 单个id
        diff_prop = max(pd.notnull(ge2_df[prop_shift]) \
                        & sim_fn(ge2_df[prop], ge2_df[prop_shift], pct) \
                        & pd.notna(ge2_df['income_tm'] - ge2_df['income_tm'].shift(1)) \
                        & (ge2_df['income_tm'] - ge2_df['income_tm'].shift(1) < pd.to_timedelta(days, 'D')))
        rst = {ge2_df['id'].unique()[0]: 1 if diff_prop else 0}
        if len(df['id'].unique())>1:
            rst.update({i: 0 for i in df['id'].unique() if i != ge2_df.loc[0, 'id']})
        result = {'data': rst, 'rst': 1}
        logger.info('query result: {} only 1 id has >1 income'.format(result))
        return result

    # 多个id
    diff_prop = ge2_df.groupby('id') \
        .apply(lambda x: pd.notnull(x[prop_shift]) \
                         & sim_fn(x[prop], x[prop_shift], pct) \
                         & pd.notna(x['income_tm'] - x['income_tm'].shift(1)) \
                         & (x['income_tm'] - x['income_tm'].shift(1) < pd.to_timedelta(days, 'D'))
               ) \
        .groupby('id').agg(max)

    diff = diff_prop[diff_prop].index.values
    diff_dict = {}
    if len(diff) > 0:
        diff_dict = dict(zip(diff, [1]*len(diff)))

    same_df = df['id'][df['id'].isin(diff) == False].unique()
    if len(same_df) > 0:
        same_dict = dict(zip(same_df, [0]*len(same_df)))
        diff_dict.update(same_dict)
    result = {'data': diff_dict, 'rst': 1}
    logger.info('query result: {}'.format(result))
    return result


if __name__ == '__main__':
    ids = ["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88="]
    # props = ['profession_station', 'start_work_tm']
    # ids = ["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88="]
    r = query_diff(ids, 'resident_city', 10, 0.9, sim_fn='jw')
    print(r)
