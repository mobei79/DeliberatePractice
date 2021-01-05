# -*- coding: utf-8 -*-
"""
@Time     :2020/11/11 14:00
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import datetime
import re
import sys
import os
import stat
import subprocess
import logging
import time


class HiveTask:
    def __init__(self,argv=[]):
        self.date_today = datetime.datetime.now().date()
        date_yesterday = self.date_today - datetime.timedelta(days=1)
        self.data_today = date_yesterday
        self.date_type = 'day'
        get_date = ''
        bushu_date = ''
        self.partition_dt = self.date_today - datetime.timedelta(days=1)
        self.stat_date = self.date_today - datetime.timedelta(days=1)

        # print(self.date_today,self.data_today,self.partition_dt,self.stat_date)
        if argv:
            import getopt
        if self.date_type not in ('day', 'week', 'month', 'monthly', 'quarter', 'quarterly', 'year', 'yearly'):
            print('日期类型参数不正确')
            print('日期类型：day, week, month, monthly, quarter, quarterly, year, yearly')
            sys.exit(1)
        self.partition_value = self.data_today

    # def get_data(self,sql):
    #     # self.logger.info(sql)
    #     try:
    #         res = self.
    #
    # def __exec_hive_sql(self,sql,data_file=None):
    #     total_write = 0
    def find_max_date(self,dbname,tbname):
        # hql = "show partitions %(dbname)s.%(tbname)s"%{'dbname':dbname,'tbname':tbname}
        hql = 'show partitions {dbname}.{tbname}'.format(dbname =dbname,tbname=tbname)
        cmd = 'hive -e "{hql}"'.format(hql=hql)
        print("cmd",cmd)
        res = os.popen(cmd).readlines()
        date_list = []
        print("res",res)
        for r in res:
            print("rrr",r)
            date_list.append(r.strip('\r\n').split('=')[1])
        date_list.sort(reverse=True)
        print(date_list)
        # for date in date_list:
        #     hql = "select 1 from %(dbName)s.%(tbName)s where dt='%(date)s' limit 1" % {'dbName': dbname,  'tbName': tbname, 'date': date}
        #     cmd = 'hive -e "%s"' % hql
        # res = os.popen(cmd).readlines()
        # if not [] == res:
        #     return date
        return 'error'

    def exec_sql(self,hql):
#          logger
        try:
            res = self.__exec_hive_sql(sql = hql)
            print(res)
            if res[0] != 0:
                raise Exception('Please Check SQL...')

            if res[1] == '0':
                print("tsttasd")
                # self.logger.error('You get a NULL table...')
        except Exception as e:
            raise e

    def __exec_hive_sql(self,sql,data_file=None):
        total_write = 0
        cmd = 'hive -e "{hql}"'.format(hql=sql)
        print(cmd)
        if data_file:
            cmd ='hive -e "{hql}" > {data_file}'.format(hql=sql,data_file=data_file)
        res = self.run_shell_cmd(cmd)
        if res[0] != 0:
    #         logger
            time.sleep(30)
            res = self.run_shell_cmd(cmd)
        print("res[1]",res[1])
        for line in res[1].split('\n'):
            print("liny",line)
            # if line.find('HDFS Read:') != 1:
            #     line = line.strip().split()
            #     change = line[line.index('HDFS'):-1]
            #     total_write = int(change[-1])

        return [res[0], total_write]

    def run_shell_cmd(self,shell_cmd,encoding='utf8',logger=True,exception_pass=False):
        res = subprocess.Popen(shell_cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        results = []
        reLogPattern = re.compile(r'^SLF4J:.*')
        while True:
            line = res.stdout.readline().decode(encoding).strip()
            if line =='' and res.poll() is not None:
                break
            elif re.match(reLogPattern,line):
                pass
            else:
                results.append(line)
                # if logging:
                #     logger
        return_code = res.returncode
        if return_code != 0 and not exception_pass:
            raise Exception('\n'.join(results))
        return [return_code,'\n'.join(results)]


if __name__=="__main__":
    # example
    exam_hivetask = HiveTask()
    hql_1 = "show partitions app.app_Knowledge_map_interface"
    hql_2 = 'hive -e "show partitions app.app_Knowledge_map_interface"'
    hql_3 = "select * from app.app_Knowledge_map_interface limit 10"
    a = exam_hivetask.exec_sql(hql_3)
    # a =exam_hivetask.run_shell_cmd(hql_2)
    # a = exam_hivetask.find_max_date('app','app_Knowledge_map_interface')
    print(a)