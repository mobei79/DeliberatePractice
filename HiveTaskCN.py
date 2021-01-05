# -*- coding: utf-8 -*-
"""
@Time     :2020/11/13 14:05
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import os
import sys
import datetime

# lib_path = os.path.abspath(os.curdir+"/")
import time

HIVE_ADDRESS=""
HIVE_PORT=""

class HiveTask:

    def __init__(self,argv=[]):
        self.date_today = datetime.datetime.now().date()
        self.partition_dt = self.date_today - datetime.timedelta(days=1)
        self.sleepSecond = 60
        self.totalSecond = 0
    def check_sql_with_fix(self,hql,compare_value,compare_type='eq'):

        rst_value = self.get_data_JDBC(hql)
        """
        检查是否是昨天的分区
        """

        """
        不检查分区，直接从分区中取数
        """
    def get_conn_neo4j(self,uri, user, password):
        from neo4j.v1 import GraphDatabase
        neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        return neo4j_driver
    def get_conn_impala(self,host=None,port=None
                        ,auth_mechanism='GSSAPI'
                        ,kerberos_service_name = 'hive'
                        ,use_kerberos = True):
        from impala.util import as_pandas
        from impala.dbapi import connect
        conn = connect(
            host=host,
            port=port,
            auth_mechanism=auth_mechanism,
            kerberos_service_name=kerberos_service_name,
            use_kerberos=use_kerberos)
        return conn
    def get_conn_pyHive(self,host=None,port=None
                        ,auth_mechanism='GSSAPI'
                        ,kerberos_service_name = 'hive'
                        ,use_kerberos = True):
        from pyhive import hive
        conn = hive.connect(host='fdw3.fengjr.inc'
                            ,port = 10000
                            ,auth = 'KERBEROS'
                            ,kerberos_service_name = 'hive')
        return conn



    def exec_sql(self,sql):
        conn = self.get_conn_pyHive()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            data = cursor.fetchall()
        except Exception as e:
            raise e
        finally:
            cursor.close()
            conn.close()


    def get_data_JDBC(self,hql,compare_num=500000):
        # logger
        conn = self.get_conn_pyHive()
        cursor = conn.cursor()
        try:
            cursor.execute(hql)
            data = cursor.fetchall()
            if len(data) >= compare_num:
                os.system("""curl -d "mobileNum=""" + str(
                    "19910631579") + """&content=""" + "图数据库N+1增量数据过多" + """&smsType=dsj-baojing" http://10.10.52.180:16680/voice/sendvoice""")
                raise EOFError("数据量过大，超过65535行，请手动导出数据")
            else:
                return data
        except Exception as e:
            raise e
        finally:
            cursor.close()
            conn.close()

    def find_max_partition_dt(self,db="app"
                              ,tb = "app_Knowledge_map_interface"
                              ,dt_time=None):
        """
        输入 库名表名
        :param db:
        :param tb:
        :param dt_time:
        :return:
        """
        hql = """show partitions {dbname}.{tbname}""".format(dbname=db,tbname=tb)
        conn = self.get_conn_impala()
        cursor = conn.cursor()
        try:
            cursor.execute(hql)
            data = cursor.fetchall()
            # if len(data) >= 65535:
            #     raise EOFError("数据量过大， 超过65535行，请手工导出数据")
            # return data
            partition_list = []
            for item in data:
                partition_list.append(item.strip())
            partition_list.sort(reverse=True)
            # print(partition_list)
            return partition_list[0]
        except Exception as e:
            raise e
        finally:
            cursor.close()
            conn.close()





