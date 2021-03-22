# -*- coding: utf-8 -*-
"""
@Time     :2021/1/20 15:58
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc : regular Expression:
    https://zhuanlan.zhihu.com/p/338826624
"""
import re
import os
import sys
"""
邮箱
包含大小写字母，下划线，阿拉伯数字，点号，中划线
"""
pattern = re.compile(r"[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)")  # 编辑正则表达式，生成正则表达式（pattern）对象
strs = '我的私人邮箱是zhuwjwh@outlook.com，公司邮箱是123456@qq.org，麻烦登记一下？'
result = pattern.findall(strs)
print(result)

"""
身份证
地区： [1-9]\d{5} 
年的前两位： (18|19|([23]\d)) 1800-2399
年的后两位： \d{2}
月份： ((0[1-9])|(10|11|12))
天数： (([0-2][1-9])|10|20|30|31) 闰年不能禁止29+
三位顺序码： \d{3}
两位顺序码： \d{2}
校验码： [0-9Xx]
"""
pattern = re.compile(r"[1-9]\d{5}(?:18|19|(?:[23]\d))\d{2}(?:(?:0[1-9])|(?:10|11|12))(?:(?:[0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]")
strs = '小明的身份证号码是130181198910235163，手机号是13987692110'
result = pattern.findall(strs)
print(result)

"""
国内手机号码
手机号都为11位，且以1开头，第二位一般为3、5、6、7、8、9 ，剩下八位任意数字
"""
pattern = re.compile(r"1[356789]\d{9}")
strs = '小明的手机号是13987692110，你明天打给他'
result = pattern.findall(strs)
print(result)

"""
ip地址
IP地址的长度为32位(共有2^32个IP地址)，分为4段，每段8位，用十进制数字表示
每段数字范围为0～255，段与段之间用句点隔开　
"""
pattern = re.compile(r"((?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d))")
strs = '''请输入合法IP地址，非法IP地址和其他字符将被过滤！
增、删、改IP地址后，请保存、关闭记事本！
192.168.8.84
192.168.8.85
192.168.8.86
0.0.0.1
256.1.1.1
192.256.256.256
192.255.255.255
aa.bb.cc.dd'''
result = pattern.findall(strs)
print(result)

"""
常见日期格式：yyyyMMdd、yyyy-MM-dd、yyyy/MM/dd、yyyy.MM.dd
"""
pattern = re.compile(r"\d{4}(?:-|\/|.)\d{1,2}(?:-|\/|.)\d{1,2}")
strs = '今天是2020/12/20，去年的今天是2019.12.20，明年的今天是2021-12-20'
result = pattern.findall(strs)
print(result)