# -*- coding: utf-8 -*-
"""
@Time     :2021/12/2 11:28
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from pyspark import SparkConf,SparkContext

# 创建SparkConf：设置的是Spark相关参数信息
# conf = SparkConf().setMaster("local[2]").setAppName("local test")
# # 创建SparkContext
# sc = SparkContext(conf=conf)
# 关闭
# sc.stop()

if __name__ == "__main__":
    conf = SparkConf().setMaster("local[2]").setAppName("local test")
    # 创建SparkContext
    sc = SparkContext(conf=conf)
    rdd1 = sc.parallelize([1,2,3,4,5,7,9]).map(lambda x:x+1)
    print(rdd1.collect())

    rdd2 = sc.parallelize(["dog","tiger","cat","tiger","tiger","cat"]).map(lambda x: (x,1)).reduceByKey(lambda x,y:x+y)
    print(rdd2.collect())

    sc.stop()