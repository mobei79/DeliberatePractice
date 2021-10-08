# -*- coding: utf-8 -*-
# !/usr/bin/env python

import sklearn as sklearn
import random,os,logging
import numpy as np
import pandas as pd
# import xgboost as xgb
import datetime

from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.externals import joblib 

MODEL_DATA_PATH = 'data_for_model'
DATA_PATH = 'data'
MODEL_PATH = 'config/model'
LOG_PATH='log'
OUTPUT_PATH = 'output'
cur_date = datetime.date.today()
cur_date_str = cur_date.strftime("%Y-%m-%d")

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(LOG_PATH+os.sep+"estimator.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_data(data_file,raw_data_file):
    data = pd.read_csv(data_file)
    data.reset_index(drop=True, inplace=True)
    raw_data = pd.read_csv(raw_data_file)
    raw_data.reset_index(drop=True, inplace=True)

    # 合并woe数据和原始表, woe表的字段加后缀
    col_names = raw_data.columns.tolist()
    rename_map = {}
    for col_name in col_names:
        new_name = col_name+'_woe'
        rename_map[col_name]=new_name
    data.rename(columns=rename_map, inplace=True)
    merge_data = pd.merge(data,raw_data,left_index=True,right_index=True)

    # 原始数据和input数据
    raw_data = merge_data[col_names]
    input_data = merge_data.drop(col_names, axis=1)

    logger.info("input data features name: %s" % ','.join(input_data.columns.tolist()))
    logger.info("input data shape: [%d,%d]" % input_data.shape)

    return input_data, raw_data, input_data.columns.tolist()


def main(model_name,data_type,result_name, file_name):  
    
    date_suffix = '_'.join(file_name.split('.')[0].split('_')[-2:])
    #date_suffix = 'testdata'
    data_file = MODEL_DATA_PATH+os.sep+data_type+'_woe_data_'+date_suffix+'.csv'
    raw_data_file = DATA_PATH+os.sep+data_type+'_fea_'+date_suffix+'.csv'
    #data_file = MODEL_DATA_PATH+os.sep+'first_overdue_test0710.csv'
    #raw_data_file = '../data/selc_first_overdue_data.csv'
    #date_suffix = '0710test'

    logger.info('load data from: %s' % data_file)
    logger.info('load raw data from: %s' % raw_data_file)
    input_data, raw_data, columns = load_data(data_file, raw_data_file)

    sc = StandardScaler()
    sc.fit(input_data)
    input_data = sc.transform(input_data)

    model_file = MODEL_PATH+os.sep+model_name
    logger.info(model_file)
    clf = joblib.load(model_file)
    pred_result = clf.predict(input_data)
    #pred_prob = clf.predict_proba(input_data)
    raw_data['overdue_pred'] = pred_result
    #raw_data['pred_prob'] = pred_prob[:,0]
    raw_data.to_csv(OUTPUT_PATH+os.sep+result_name+'_'+date_suffix+'.csv')
    logger.info('predict done.')    


if __name__ == "__main__":

    main('first_period_lr_woe_model_v1.pkl','first_period','first_period_predict_result_lr_v1','raw_feature_2020-07-22_2020-07-27.csv')
    #main('first_period_dt_woe_model_v1.pkl','first_period','first_period_predict_result_dt_v1','raw_feature_2020-07-22_2020-07-27.csv')
    #main('first_period_svm_woe_model_v1.pkl','first_period','first_period_predict_result_svm_v1','raw_feature_2020-07-22_2020-07-27.csv')
    #main('first_period_svm_woe_model_v1.pkl','first_period','first_period_predict_result_svm_v1','raw_feature_2020-07-23_2020-07-28.csv')
    #main('first_overdue_lr_woe_model_v1.pkl','first_overdue','first_overdue_predict_result_lr_v1','raw_feature_2020-07-23_2020-07-28.csv')
    #main('first_overdue_dt_woe_model_v1.pkl','first_overdue','first_overdue_predict_result_dt_v1','raw_feature_2020-07-23_2020-07-28.csv')
    # main('first_overdue_svm_woe_model_v3.pkl','first_overdue','first_overdue_predict_result_svm','raw_feature_2020-07-22_2020-07-27.csv')
    #main('his_overdue_lr_woe_model_v1.pkl','his_overdue','his_overdue_predict_result_lr_v1','raw_feature_2020-07-22_2020-07-27.csv')
    #main('his_overdue_lr_woe_model_v1.pkl','his_overdue','his_overdue_predict_result_lr_v1','raw_feature_2020-07-23_2020-07-28.csv')
    #main('his_overdue_dt_woe_model_v1.pkl','his_overdue','his_overdue_predict_result_dt','raw_feature_2020-07-22_2020-07-27.csv')
