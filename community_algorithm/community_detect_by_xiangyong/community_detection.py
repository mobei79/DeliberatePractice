# -*- coding: utf-8 -*-
"""
@Time     :2021/9/7 16:29
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
# !/usr/bin/env python
# coding=utf-8
# creator:xiangyong.yang@fengjr.com
import sys
import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community

lib_path = os.path.abspath(os.curdir + '/../sbin')  # 调用一个本地的包，首先向上一层，到sbin目录下
sys.path.append(lib_path)
from HiveTask import HiveTask  # 调用本地的HiveTask包，

ht = HiveTask(sys.argv[1:])  # ht存储参数列表，从第一个到最后一个。


def readDataFromHive(hql, filePath):  # 从hive中读取参数
    try:
        cmd = 'hive -e "%(hql)s" > %(filePath)s' % {'hql': hql,
                                                    'filePath': filePath}  # 使用sql从表中查找出数据，存储到本地的filepath文件中。PS hive - e是一个hive用法
        print(cmd)  # 打印之
        print
        'if file %s exist then remove it...' % filePath  # 如果路径存在，删除，然后重新存储。保证数据是最新的。
        if os.path.exists(filePath):
            os.remove(filePath)
        print
        'read data from hive and save it in localfile %s...' % filePath
        os.system(cmd)
        return cmd
    except Exception as e:
        raise e


def writeDataToHive(filePath, dbName, tbName, partition):
    try:
        hql = "load data local inpath '%(filePath)s' overwrite into table %(dbName)s.%(tbName)s partition(%(partition)s)" % {
            'filePath': filePath, 'dbName': dbName, 'tbName': tbName, 'partition': partition}
        cmd = 'hive -e "%s"' % hql  # 上一句是从本地文件系统加载数据
        print
        cmd
        print
        'load data to hive from localfile %s...' % filePath
        os.system(cmd)
        return cmd
    except Exception as e:
        raise e


if __name__ == '__main__':
    dbName = 'mining'  # 数据库名称
    tbName = 'dm_loan_usr_cluster_d'  # 表的名称
    end_date = ht.partition_value  # 调用本地包中的方法
    partition = "dt='%s'" % end_date  # 拼接
    print
    'dbName=%s' % dbName, 'tbName=%s' % tbName, 'end_date=%s' % end_date, 'partition=%s' % partition
    hql = """
			select
				s.from_id
				,s.to_id
			from 
				(
					select 
						user_id
					from 
						dwi.dwi_user_feature_lend_detail_day
					where 
						dt='%(end_date)s'
						and apply_date>='2017-10-11'
						and apply_date<='%(end_date)s'
						and is_expire=1
						and overdue_days>0
					group by 
						user_id
					union all 
					select
						user_id
					from 
						dwi.dwi_usr_special_list_full
					where 
						dt='%(end_date)s'
						and user_id is not null
						and is_court+is_black+is_overdue>0
					group by 
						user_id
				) t 
			join 
				(
					select
						*
					from 
						dwi.dwi_user_feature_relation_day
					where 
						dt='%(end_date)s'
						and length(from_id)>5
						and length(to_id)>5
				) s 
			on 
				t.user_id=s.from_id
			where 
				(s.type=3)
				or (s.type=4 and s.from_id<>s.to_id)
				or (s.type=7 and weight>=5)
				or (s.type=8 and weight>=3)
				or (s.type=1 and weight>=1)
			group by 
				s.from_id
				,s.to_id
		""" % {'end_date': end_date}
    readDataFromHive(hql, os.getcwd() + '/data.csv')

    delimiter = '\001'
    hql = """
			use %(dbName)s;
			create table if not exists %(tbName)s 
			(
				user_id		 	   	string	
				,cluster            int 
			) 
			partitioned by (dt string comment 'dt')
			row format delimited 
			fields terminated by '%(delimiter)s'
			stored as textfile
		""" % {'dbName': dbName, 'tbName': tbName, 'delimiter': delimiter}
    ht.exec_sql(hql)

    graph = nx.read_edgelist(os.getcwd() + '/data.csv', delimiter='\t', data=(('weight', float),), encoding='utf-8')
    print('nodes=%s' % graph.number_of_nodes(), 'edges=%s' % graph.number_of_edges())
    best_partition = community.best_partition(graph)
    # print partition
    # print set(partition.values())
    res = pd.DataFrame(list(best_partition.iteritems()), columns=['id', 'cluster'])
    # print res
    print
    res.groupby(['cluster']).count()
    res.to_csv(os.getcwd() + '/result.csv', sep=',', encoding='utf-8', header=False, index=False)
    writeDataToHive(os.getcwd() + '/result.csv', dbName, tbName, partition)

    resTbName = 'dm_loan_usr_community_risk_list_d'
    hql = """
			use %(dbName)s;
			create table if not exists %(resTbName)s 
			(
				id_card_md5		 	   	string	
				,login_phone_md5        string 
			) 
			partitioned by (dt string comment 'dt')
			row format delimited 
			fields terminated by '%(delimiter)s'
			stored as textfile
		""" % {'dbName': dbName, 'resTbName': resTbName, 'delimiter': delimiter}
    ht.exec_sql(hql)

    hql = """
			use %(dbName)s;
			insert overwrite table %(resTbName)s partition(dt='%(end_date)s')
			select
				w.id_card_md5
				,w.login_phone_md5
			from 
				(
					select
						t.cluster
						,count(1) as cnt 
						,sum(case when nvl(s.is_pass,0)=0 then 1 else 0 end) as refuse_cnt 
						,sum(case when nvl(s.is_pass,0)=0 then 1 else 0 end)/count(1) as refuse_rate

						,sum(nvl(s.is_expire,0)) as expire_cnt 
						,sum(case when s.overdue_days>0 then 1 else 0 end) as overdue_cnt 
						,sum(case when s.overdue_days>0 then 1 else 0 end)/sum(nvl(s.is_expire,0)) as overdue_rate 
					from 
						(
							select
								*
							from 
								%(dbName)s.%(tbName)s
							where 
								dt='%(end_date)s'
						) t 
					left join 
						(
							select
								user_id
								,max(is_pass)      as is_pass
								,max(is_expire)    as is_expire
								,max(overdue_days) as overdue_days
							from 
								dwi.dwi_user_feature_lend_detail_day
							where 
								dt='%(end_date)s'
								and apply_date>='2017-10-11'
								and apply_date<='%(end_date)s'
							group by 
								user_id
						) s 
					on  
						t.user_id=s.user_id
					group by 
						t.cluster
				) t 
			join 
				(
					select
						*
					from 
						%(dbName)s.%(tbName)s
					where 
						dt='%(end_date)s'
				) s 
			on 
				t.cluster=s.cluster
			join 
				(
					select 
						user_id
						,login_phone_md5
						,id_card_md5
					from 
						dwd.dwd_user_borrower_user_account_full
					where 
						dt='%(end_date)s'
				) w 
			on 
				s.user_id=w.user_id
			where 
				t.refuse_rate>=0.7
				and t.overdue_rate>=0.7
			group by 
				w.id_card_md5
				,w.login_phone_md5
		""" % {'dbName': dbName, 'tbName': tbName, 'end_date': end_date, 'resTbName': resTbName}
    ht.exec_sql(hql)

    hql = """
			insert overwrite table mining.dm_usr_cluster_mobile_cache_full
			select
				edw_user_encrypt(login_phone_md5,'qwertyuiop') as key
				,edw_user_encrypt(
                toJson
                    (
                        'login_phone_md5',login_phone_md5,0
                    ),
                'qwertyuiop'
            ) as json
			from 
				%(dbName)s.%(resTbName)s
			where 
				dt='%(end_date)s' 
				and login_phone_md5 is not null               
		""" % {'dbName': dbName, 'end_date': end_date, 'resTbName': resTbName}
    ht.exec_sql(hql)


