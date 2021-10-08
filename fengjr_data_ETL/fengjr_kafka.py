# -*- coding: utf-8 -*-
"""
@Time     :2021/9/7 15:04
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from kafka import KafkaProducer
import gzip
import StringIO

def gzip_compress(msg_str):
    try:
        buf = StringIO.StringIO()
        with gzip.GzipFile(mode='wb', fileobj=buf) as f:
            f.write(msg_str)
        return buf.getvalue()
    except BaseException as e:
        print("Gzip压缩错误" + e)

def gzip_uncompress(c_data):
    try:
        buf = StringIO.StringIO(c_data)
        with gzip.GzipFile(mode='rb', fileobj=buf) as f:
            return f.read()
    except BaseException as e:
        print ("Gzip解压错误" + e)

def sned(topic_name, msg, key=None):
    if key is not None:
        producer = KafkaProducer(bootstrap_servers=['10.10.202.19:9092', '10.10.202.21:9092'],
                                 key_serializer=gzip_compress,
                                 value_serializer=gzip_compress)
        r = producer.send(topic_name, value=msg, key=key)
    else:
        producer = KafkaProducer(bootstrap_servers=['10.10.202.19:9092', '10.10.202.21:9092'],
                                 value_serializer=gzip_compress)
        r = producer.send(topic_name, value=msg)
    # producer.flush(timeout=5)
    producer.close(timeout=5)
    return r

if __name__ == "__main__":
    str = "test kafka python"
    topic = "test"
    print(send(topic, str, key="fengjr").value)