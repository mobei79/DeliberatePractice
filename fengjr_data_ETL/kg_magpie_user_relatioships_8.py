# -*- coding: utf-8 -*-
"""
@Time     :2020/12/1 15:49
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import sys
import os
from HiveTaskCN import HiveTask
ht = HiveTask(sys.argv[1:])
def create_table(dbname:str,tbname:str,delimiter:str):
    print(f'create table:\ndbname={dbname}\ntbname={tbname}')
    hql=f"""
    use {dbname};
    drop table if exists {tbname};
    create external table {tbname}(
        type     int    comment '关系类型'
        ,from_id string comment 'from node'
        ,to_id   string comment 'end node'
        ,weight  int    comment '权重'
    ) comment '{tbname} 类型8关系表'
    partitioned by (dt  string  comment '分区')
    row format delimited fields terminated by '{delimiter}'
    stored as textfile;
    """
    ht.exec_sql(hql)

def insert_table(dbname:str,tbname:str):
    dt =ht.partition_dt
    hql = f"""
    use {dbname};
    insert overwrite tabel {tbname} partition(dt='{dt}')
    
    """

if __name__ == "__main__":
    assert len(sys.argv)>1 and len(sys.argv) != 2,'输入的参数长度不符合要求，请检查是否包含 库 表 分隔符！'
    dbname = sys.argv[1]
    tbname = sys.argv[2]
    delimiter='\001'
    print(f'dbname={dbname}\ntbname={tbname}\ndelimiter={delimiter}\nbegin ... ...')
    create_table(dbname,tbname,delimiter)
    insert_table(dbname,tbname)
    print('success ... ...')