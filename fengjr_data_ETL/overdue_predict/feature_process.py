# -*- coding: utf-8 -*-
"""
@Time     :2021/8/31 14:00
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import pandas as pd
#from sklearn.impute import SimpleImputer
from data.feature_config import *
import numpy as np
# from fengjr_overdue_predict.overdue_predict_scheduler.config.feature_config import *
# from config.feature_config import *
import os, logging, pprint
import json,chardet
import datetime

'''

设定目标： 提前多少天预测还款逾期，即抽取应还日期
app_log数据回溯时间天数 15天
项目目标：根据用户现在的基本特征和15天内的行为特征，预测7天后是否会逾期


0 预测：
    方法1：
        抽取七天后到期的用户，得到用户的基础特征和15天内的行为日志；
        7天后根据逾期情况，训练模型；
    方法2：
        抽取昨天到期的用户，得到行为基础特征，7天前15天内的行为日志，根据用户是cleared或者overdue进行预测；
    
细节点：
    线上离线计算时，抽取账单日为7天后，且未逾期的用户


1 首先是从业务数据中抽取样本数据；【因为实验都是在样本上进行的，所以先取样本，尽量覆盖全面】
    1.1 怎么取？
        抽取待还标的样本数据：dm_user_overdue_predict_sample【也就是处理那些用户的数据】
            从账务核心还款计划表中抽取七天后到期的标的，至临时表；【due_time为七天后；fact_time is NULL；plan_status=UNDUE未逾期；periods_no!=0期数不为0的】
        抽取样本特征：
            【用上表dm_user_overdue_predict_sample 去关联其他的业务表，得到用户基本特征，
                保存到临时表dm_user_overdue_predict_feature_tmp  //  or offline；
                】
            【用上表dm_user_overdue_predict_sample 去关联app埋点流量表 取样本用户15天内的行为日志，
                保存到临时表dm_user_overdue_user_app_log_tmp 】
                用户行为日志：包含ip uniqid visit_times channel platform、log_type(click/view.)、cookie_id、访问时间、seq_num
                    主要是：ct_page 和 click_name（点击id例如：repayment_repaylist_check,apiRequest等）、页面描述
                
    1.2 怎么处理？
        导出dm_user_overdue_user_app_log_tmp 用户行为日志数据。
            导出列名 desc table + 导出数据，然后组合dataform
            将数据映射为{userId:[[log1, log2, log3,....]]}
                因为我们主要看日志中的 userId的ct_page和click_name，要将这些埋点名称映射为编号；
                然后统计每个用户的行为统计数据；返回一个DataForm结果；
            return app_log_Dataform
        导出dm_user_overdue_predict_feature_tmp 用户基础特征数据
            导出列名 desc table + 导出数据，然后组合dataform
        将基础特征和行为日志特征 按照userId进行组合。
    
    1.3 将数据存储到指定目录。将文件名称存储到特定的文件，用于后续预测的访问；
2 特征工程
    样本拆分：
        将样本按照类别拆分为：首期待还，无历史逾期，其他待还(非首期且有逾期历史)
    样本筛选： 去除期数为0的样本
    特征筛选： 选择训练中效果好的特征
    中间保存特征筛选后的数据
    特征处理： 日期特征转换；缺失值补充；特征分箱（读取边界，然后分箱）；
    读取woe table表
    特征woe编码；
    保存woe格式数据

3 预测
    读取模型预测；
    
'''
def feature_extract():
    #1 抽取当日处理样本相关特征262项 到临时表 dm_user_overdue_predict_feature_tmp
    #2 抽取当日样本关联的applog日志400余项行为类别 到临时表 dm_user_overdue_user_app_log_tmp

    #2.2 导出列名和数据；
    #3 合并两部分数据；
    return 1
def  del_sample_custom(sample_data, feature_name, value_range):
    if len(value_range)==2:
        result_data = sample_data[(sample_data[feature_name]<value_range[0]) | (sample_data[feature_name]>feature_name[1])]
    if len(value_range)==1:
        result_data = sample_data[sample_data[feature_name]!=value_range[0]]
    return result_data



def feature_processing(raw_feature_data_file):
    date_suffix = '_'.join(raw_feature_data_file.split('.')[0].split('_')[-2:])
    data_suffix = '_'.join(raw_feature_data_file.split(".")[0].split('_')[-2:])
    print(data_suffix)
    raw_data = pd.read_csv(raw_feature_data_file)
    print(raw_data.shape)

    raw_data.drop(raw_data[raw_data.plan_status.isin(["PRE_CLEARED"])].index, inplace=True)  # 去除提前还款的标的
    first_period_data = raw_data[raw_data['periods_no'] == 1]
    first_overdue_data = raw_data[(raw_data['periods_no'] != 1) & (raw_data['his_overdue_cnt'] == 0)]
    his_overdue_data = raw_data[(raw_data['periods_no'] != 1) & (raw_data['his_overdue_cnt'] != 0)]
    print('first_period_data data shape: [%d, %d]' % first_period_data.shape)
    print('first_overdue_data data shape: [%d, %d]' % first_overdue_data.shape)
    print('his_overdue_data data shape: [%d, %d]' % his_overdue_data.shape)

    # step3: 样本筛选
    merge_data_selc = del_sample_custom(raw_data, 'periods_no', [0]) # 去除期数为0的样本

    # step3: 特征筛选
    first_period_data_selc = pd.DataFrame()
    first_overdue_data_selc = pd.DataFrame()
    his_overdue_data_selc = pd.DataFrame()
    first_period_fea_select_dict = first_overdue_feature_select_v1
    first_overdue_fea_select_dict = first_overdue_feature_select_v1
    his_overdue_fea_select_dict = his_overdue_feature_select_v1

    for fea_name in first_period_fea_select_dict['selected']:
        first_period_data_selc[fea_name] = first_period_data[fea_name]
    for fea_name in first_overdue_fea_select_dict['selected']:
        first_overdue_data_selc[fea_name] = first_overdue_data[fea_name]
    for fea_name in his_overdue_fea_select_dict['selected']:
        his_overdue_data_selc[fea_name] = his_overdue_data[fea_name]

    first_period_data_selc.drop(['plan_status'], axis=1, inplace=True)
    first_overdue_data_selc.drop(['plan_status'], axis=1, inplace=True)
    his_overdue_data_selc.drop(['plan_status'], axis=1, inplace=True)
    first_period_data_selc.reset_index(drop=True, inplace=True)
    first_overdue_data_selc.reset_index(drop=True, inplace=True)
    his_overdue_data_selc.reset_index(drop=True, inplace=True)
    print('first_period_data_selc shape: [%d, %d]' % first_period_data_selc.shape)
    print('first_overdue_data_selc shape: [%d, %d]' % first_overdue_data_selc.shape)
    print('his_overdue_data_selc shape: [%d, %d]' % his_overdue_data_selc.shape)

    DATA_PATH = "D:\Program Files\pythonwork\DeliberatePractice\\fengjr_data_ETL\overdue_predict\data"
    # step4: 保存特征筛选后的数据
    out_file_1 = DATA_PATH + os.sep + 'first_period_fea_%s.csv' % date_suffix
    out_file_2 = DATA_PATH + os.sep + 'first_overdue_fea_%s.csv' % date_suffix
    out_file_3 = DATA_PATH + os.sep + 'his_overdue_fea_%s.csv' % date_suffix
    first_period_data_selc.to_csv(out_file_1, index=None)
    first_overdue_data_selc.to_csv(out_file_2, index=None)
    his_overdue_data_selc.to_csv(out_file_3, index=None)
    print('output data file: %s' % out_file_1)
    print('output data file: %s' % out_file_2)
    print('output data file: %s' % out_file_3)
    first_period_data_selc = first_period_data_selc.drop(['user_id', 'loan_id'], axis=1, inplace=False)
    first_overdue_data_selc = first_overdue_data_selc.drop(['user_id', 'loan_id'], axis=1, inplace=False)
    his_overdue_data_selc = his_overdue_data_selc.drop(['user_id', 'loan_id'], axis=1, inplace=False)

if __name__ == "__main__":
    path = "D:\Program Files\pythonwork\DeliberatePractice\\fengjr_data_ETL\overdue_predict\data\\raw_feature_2021-09-22_2021-09-27.csv"
    feature_processing(path)