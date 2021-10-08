-- 应用举例，查找页面点击量变化趋势
select
count(*)
  --substr(request_tm,1,13) as request_tm,count(*) as cnt
from
  dwi.dwi_flow_click_log
where
  ds='app' and dt='2020-11-24' and ct_page='HomepageFragment'
group by
  request_tm


-- create new table
use mining_magpie;
DROP table if exists mining_magpie.hive_test;
CREATE EXTERNAL TABLE mining_magpie.hive_test(
    name string COMMENT '姓名'
    ,frinds array<string> COMMENT '朋友姓名' -- gujj bingbing,lili
    ,children map<String, int> COMMENT '孩子' -- gujj namexiaosong:18,xiaoxiaosong:19
    ,address struct<cname:string, city:string, youbian:string> -- gjj huilong,beijing,100000
) COMMENT '手工建表测试'
--partitioned by (dt string COMMENT 'dt' )
row format delimited fields terminated by ','
collection items terminated by '_'
map keys terminated by ':'
lines terminated by '\n'  --行分隔符默认为\n（仅支持） 加不加都一样
stored as textfile;
LOAD DATA local INPATH '/tmp/jingjin.guo/test.txt' OVERWRITE INTO TABLE hive_test partition (dt='2020-11-26');