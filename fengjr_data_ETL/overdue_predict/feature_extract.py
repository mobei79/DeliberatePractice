# -*- coding: utf-8 -*-
# !/usr/bin/env python
import pyhs2
import pprint,csv,os,codecs
import datetime
import pandas as pd
import logging

cur_date = datetime.date.today()
cur_date_str = cur_date.strftime("%Y-%m-%d")
LOG_PATH='log'

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(LOG_PATH+os.sep+"feature_extract.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def read_sql_file(file_name):
    with open(file_name) as f:
        sql = f.read()
    return sql

DATA_PATH = 'data'
CONF_PATH = 'config'
SQL_PATH = 'hive_sql'
TABLE_NAME = 'dm_user_overdue_predict_sample'
fea1_sql = read_sql_file(SQL_PATH+os.sep+'fea1.sql')
app_log_sql = read_sql_file(SQL_PATH+os.sep+'app_log.sql')

ip = "fdw3.fengjr.inc"
port = 10000
authMechanism="KERBEROS"
conn = pyhs2.connect(host=ip, port=port, authMechanism=authMechanism)


def load_data_from_hive(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetch()
    df_data = pd.DataFrame(data)
    return df_data

def read_sql_file(file_name):
    with open(file_name) as f:
        sql = f.read()
    return sql

# 加载新增页面特征名
def load_field_map():
    page_field_list = []
    with codecs.open(CONF_PATH+os.sep+'click_name.csv', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(',')[0]
            page_field_list.append(line)
    with codecs.open(CONF_PATH+os.sep+'ct_page.csv', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(',')[0]
            page_field_list.append(line)
    return page_field_list

# 构建userid to log data 映射表
def build_user2log(all_data):
    app_log_data = {}
    for idx, row in all_data.iterrows():
        data=row.to_dict()
        log_data_item = {}
        for k,v in data.items():
            k = k.split('.')[-1]
            log_data_item[k]=v
        user_id = log_data_item["user_id"]
        if user_id in app_log_data:
            app_log_data[user_id].append(log_data_item)
        else:
            app_log_data[user_id] = [log_data_item]
    return app_log_data

# app_log 衍生特征计算
def join_and_stat(user2log, page_field_list):
    out_df = pd.DataFrame() 
    for user_id, log_list in user2log.items():
        new_item = {}
        new_item['user_id'] = user_id
        new_item['first_visit_time']=None
        new_item['last_visit_time']=None
        new_item['last_device_name']=None
        new_item['visit_times']=0
        new_item['d15_visit_times']=0
        new_item['d15_click_times']=0
        new_item['d15_view_times']=0
        for key in page_field_list:
            new_key = 'd15_'+key+'_count'
            new_item[new_key]=0

        first_visit_time = ''
        last_visit_time = ''
        last_device_name = ''
        visit_times = 0
        d15_visit_times = 0
        d15_click_times = 0
        d15_view_times = 0

        for log_item in log_list:
            log_time = datetime.datetime.strptime(log_item['request_tm'], '%Y-%m-%d %H:%M:%S')
            first_visit_time = log_item['first_session_tm']
            if last_visit_time=='':
                last_visit_time=log_item['request_tm']
                last_device_name=log_item['device_name']
                visit_times=log_item['visit_times']
            else:
                cur_log_time =datetime.datetime.strptime(log_item['request_tm'], '%Y-%m-%d %H:%M:%S') 
                last_log_time =datetime.datetime.strptime(last_visit_time, '%Y-%m-%d %H:%M:%S') 
                if cur_log_time>last_log_time:
                    last_visit_time=log_item['request_tm']
                    last_device_name=log_item['device_name']
                    visit_times = log_item['visit_times']
            d15_visit_times+=1
            if log_item['log_type']=='click':
                d15_click_times+=1
            if log_item['log_type']=='view':
                d15_view_times+=1
            page = log_item['ct_page']
            if page in page_field_list:
                new_item['d15_'+page+'_count']+=1
            click_name = log_item['click_name']
            if click_name in page_field_list:
                new_item['d15_'+click_name+'_count']+=1
        new_item['first_visit_time']=first_visit_time
        new_item['last_visit_time']=last_visit_time
        new_item['last_device_name']=last_device_name
        new_item['visit_times']=visit_times
        new_item['d15_visit_times']=d15_visit_times
        new_item['d15_click_times']=d15_click_times
        new_item['d15_view_times']=d15_view_times

        out_df = out_df.append(pd.Series(new_item), ignore_index=True)
    return out_df

def feature_extract(cur_date, interval=7, app_log_interval=15):
    # 时间点计算，格式转换
    due_date = cur_date+datetime.timedelta(days=interval)
    dt = cur_date-datetime.timedelta(days=1)
    app_log_s_date = dt-datetime.timedelta(days=app_log_interval)

    due_date = due_date.strftime("%Y-%m-%d")
    dt = dt.strftime("%Y-%m-%d")
    cur_date = cur_date.strftime("%Y-%m-%d")
    app_log_s_date = app_log_s_date.strftime("%Y-%m-%d")
    
    logger.info("start feature extract...")
    logger.info("cur_date: %s" % cur_date)
    logger.info("predict due date: %s" % due_date)
    logger.info("feture extract date limit[dt]: %s" % dt)
    logger.info("app log feture date range: [%s, %s]" % (app_log_s_date, dt))
    
    # 抽特征1  -->tmp  抽当日处理样本相关特征，保存至临时表dm_user_overdue_predict_feature_tmp
    all_sql = fea1_sql % (TABLE_NAME, cur_date, dt,dt, dt, dt, dt,dt)
    for sql in all_sql.split(";"):
        cur = conn.cursor()
        cur.execute(sql)
    logger.info("first batch feture extract to: dm_user_overdue_predict_feature_tmp")

    # 抽当日样本关联app log，保存至临时表 dm_user_overdue_user_app_log_tmp
    sql = app_log_sql % (TABLE_NAME, cur_date, cur_date, app_log_s_date)
    cur = conn.cursor()
    cur.execute(sql)
    logger.info("app log feture extract to: dm_user_overdue_user_app_log_tmp")
    
    # 导出dm_user_overdue_user_app_log_tmp 数据
    sql_1 = "desc dm_user_overdue_user_app_log_tmp" # 导出列名
    df_tmp = load_data_from_hive(conn, sql_1)
    cols = df_tmp[0].tolist()
    sql_2 = "select * from dm_user_overdue_user_app_log_tmp" # 导出数据
    df_app_log_data = load_data_from_hive(conn, sql_2)
    df_app_log_data.columns = cols 
    logger.info("raw app log data shape: [%d, %d]" % df_app_log_data.shape)

    # app log 特征计算
    page_field_list = load_field_map()
    user_2_log = build_user2log(df_app_log_data)
    ###
    for k, v in user_2_log.items():
        print('user: '+k)
        print(len(v))
        print('\n')
    print(len(user_2_log))
    app_log_feature = join_and_stat(user_2_log, page_field_list)
    logger.info("app log feature data shape: [%d, %d]" % app_log_feature.shape)

    # 导出fea1特征
    sql_1 = "desc dm_user_overdue_predict_feature_tmp" # 导出列名
    df_tmp = load_data_from_hive(conn, sql_1)
    cols = df_tmp[0].tolist()
    sql_2 = "select * from dm_user_overdue_predict_feature_tmp" # 导出数据
    feature_data_1 = load_data_from_hive(conn, sql_2)
    feature_data_1.columns = cols 
    logger.info("first batch feature data shape: [%d, %d]" % feature_data_1.shape)

    # dm_user_overdue_predict_feature_tmp 合并app log 特征
    all_feature_data = feature_data_1.join(app_log_feature.set_index('user_id'), on='user_id', how='left')
    out_file = 'raw_feature_'+cur_date+'_'+due_date+'.csv'
    all_feature_data.to_csv(DATA_PATH+os.sep+out_file,index=None)
    with open('date_align.record','w') as f:
        f.write(out_file)
    logger.info("all raw feature data shape: [%d, %d]" % all_feature_data.shape)
    logger.info("export all raw feature data file to: %s" % DATA_PATH+os.sep+out_file)
    # 存到dm_user_overdue_predict_feature

if __name__ == "__main__":
    # print(cur_date)
    feature_extract(cur_date, days_interval, app_log_interval)
