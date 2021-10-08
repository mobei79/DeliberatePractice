#!/usr/bin/env python
#coding=utf-8
#author:xiangyong.yang@fengjr.com
from pyspark.sql import HiveContext
from pyspark import SparkContext, SparkConf
import pandas as pd
if __name__=='__main__':
	conf = SparkConf()\
				.setAppName('test')
	sc = SparkContext(conf = conf)
	hc = HiveContext(sc)
	hql="select *from mining.dm_fhjr_usr_info_td where dt='2017-12-12' limit 100"
	x=hc.sql(hql)
	print (x.count(),x.first())
	print (x.toPandas().loc[:,'user_id'])
	print (x.toPandas().iloc[0:1,:])

#spark-submit testPySpark.py --principal hdfs@hadoop_edw --keytab /data/key/hdfs.keytab --master yarn --deploy-mode client
#spark-submit --master local[1] testPySpark.py
#spark-submit testPySpark.py --principal hdfs@hadoop_edw --keytab /data/key/hdfs.keytab --master yarn --deploy-mode client
#/opt/fengjrspark/spark-2.2.1-client/bin/spark-submit --master local[4] --principal hdfs@hadoop_edw --keytab /data/key/hdfs.keytab testPySpark.py