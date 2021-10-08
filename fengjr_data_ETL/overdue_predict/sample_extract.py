# -*- coding: utf-8 -*-
# !/usr/bin/env python
import pyhs2
import pprint,csv,os
import datetime
import logging

cur_date = datetime.date.today()
cur_date_str = cur_date.strftime("%Y-%m-%d")
LOG_PATH='log'

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(LOG_PATH+os.sep+"sample_extract.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def read_sql_file(file_name):
    with open(file_name) as f:
        sql = f.read()
    return sql

TABLE_NAME = 'dm_user_overdue_predict_sample'
HIVE_SQL = read_sql_file('hive_sql'+os.sep+'sample.sql')

ip = "fdw3.fengjr.inc"
port = 10000
authMechanism="KERBEROS"
conn = pyhs2.connect(host=ip, port=port, authMechanism=authMechanism)

def sample_extract(cur_date, interval=7):
    # 样本抽取时间点配置
    due_date = cur_date+datetime.timedelta(days=interval)
    dt = cur_date-datetime.timedelta(days=1)
    due_date = due_date.strftime("%Y-%m-%d")
    dt = dt.strftime("%Y-%m-%d")
    cur_date = cur_date.strftime("%Y-%m-%d")
    
    logger.info("start sample extract...")
    logger.info("cur_date: %s" % cur_date)
    logger.info("predict due date: %s" % due_date)
    logger.info("feture extract date limit[dt]: %s" % dt)

    # 抽取7天后到期的标的，存放至表: dm_user_overdue_predict_sample
    sql = HIVE_SQL % (TABLE_NAME, cur_date, dt, due_date)
    cur = conn.cursor()
    cur.execute(sql)

    logger.info("sample extract done: %s" % TABLE_NAME)

if __name__ == "__main__":
    interval = days_interval
    sample_extract(cur_date, interval)
