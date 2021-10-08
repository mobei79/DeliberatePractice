# -*- coding: utf-8 -*-
# !/usr/bin/env python
import sys,os
import sample_extract
import feature_extract
import feature_process_pre
import estimator

import datetime
import logging
import traceback

cur_date = datetime.date.today()
#cur_date = datetime.date(2020,07,24)
cur_date_str = cur_date.strftime("%Y-%m-%d")
DATA_PATH = 'data'
LOG_PATH='log'

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(LOG_PATH+os.sep+"daily_run.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 提前多少天预测还款逾期，即抽取应还日期 为【当前日期 + 7d】 的待还标的
days_interval=5

# app_log数据回溯时间天数
app_log_interval = 15

# step1: 抽取待还标的样本数据
logger.info('start overdue predict workflow...')
try:
    sample_extract.sample_extract(cur_date, days_interval)
    logger.info('sample extract done.')
except Exception as msg:
    logger.error(msg)
    logger.error(traceback.format_exc())

# step2: 特征抽取与计算
try:
    feature_extract.feature_extract(cur_date, days_interval, app_log_interval)
    logger.info('feature extract done.')
except Exception as msg:
    logger.error(msg)
    logger.error(traceback.format_exc())

# step3: 特征工程
file_name = ''
with open('date_align.record', 'r') as f:
    file_name = f.readline().strip()
if file_name == '':
    logger.error("raw feature file not found!")
raw_data_file = DATA_PATH+os.sep+file_name
try:
    feature_process_pre.feature_process_pre(raw_data_file)
    logger.info('feature process done.')
except Exception as msg:
    logger.error(msg)
    logger.error(traceback.format_exc())

# step4: 逾期模型预测
# 模型选择
model_1 = 'first_period_lr_woe_model_v1.pkl'
output_1 = 'first_period_predict_result'
model_2 = 'first_period_lr_woe_model_v1.pkl'
output_2 = 'first_overdue_predict_result'
model_3 = 'his_overdue_lr_woe_model_v1.pkl'
output_3 = 'his_overdue_predict_result'
try:
    estimator.main(model_1, 'first_period', output_1, file_name)
    estimator.main(model_2, 'first_overdue', output_2, file_name)
    estimator.main(model_3, 'his_overdue', output_3, file_name)
    logger.info('estimate done.')
except Exception as msg:
    logger.error(msg)
    logger.error(traceback.format_exc())
logger.info('overdue predict finish.')
