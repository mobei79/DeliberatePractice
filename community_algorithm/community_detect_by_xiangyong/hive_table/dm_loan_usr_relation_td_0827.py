#!/usr/bin/env python
#coding=utf-8
#creator:xiangyong.yang@fengjr.com
import os
import sys
import datetime
import string
lib_path = os.path.abspath(os.curdir + '/../sbin')
sys.path.append(lib_path)
from HiveTask import HiveTask
ht = HiveTask(sys.argv[1:])
def getDateDelta(basetime,delta):
	#basetime is a string like '2017-01-01 01:01:01'
	d1 = datetime.datetime(string.atoi(basetime[0:4]),string.atoi(basetime[5:7]),string.atoi(basetime[8:10]))
	d2 = d1 + datetime.timedelta(days = delta)
	deltaDate = d2.strftime('%Y-%m-%d')
	print deltaDate
	return deltaDate

def find_max_date(dbName,tbName):
	hql="show partitions %(dbName)s.%(tbName)s"%{'dbName':dbName,'tbName':tbName}
	cmd='hive -e "%s"'%hql
    	print cmd
	res=os.popen(cmd).readlines()
	date_list=[]
	for r in res:
		date_list.append(r.strip('\r\n').split('=')[1])
	date_list.sort(reverse=True)
	print date_list
	for date in date_list:
		hql="select 1 from %(dbName)s.%(tbName)s where dt='%(date)s' limit 1"%{'dbName':dbName,'tbName':tbName,'date':date}
		cmd='hive -e "%s"'%hql
    	print cmd
    	res=os.popen(cmd).readlines()
    	if not [] == res:
    		return date 
    return 'error'

def createTable(dbName,tbName,delimiter):
	print 'dbName=%s'%dbName,'tbName=%s'%tbName,'delimiter=%s'%delimiter
	hql="""
			use %(dbName)s;
			create table if not exists %(tbName)s 
			(
				type     int 
				,from_id string
				,to_id   string
				,weight  int 
			) comment '%(tbName)s'
			partitioned by (dt string comment 'dt')
			row format delimited 
			fields terminated by '%(delimiter)s'
			stored as textfile
		"""%{'dbName':dbName,'tbName':tbName,'delimiter':delimiter}
	#print hql
	ht.exec_sql(hql)

def insertTable(dbName,tbName,):
	end_date=ht.partition_value
	max_date=find_max_date('dwd','dwd_user_borrower_user_account_full')
 	print 'dbName=%s'%dbName,'tbName=%s'%tbName,'end_date=%s'%end_date,'max_date=%s'%max_date
	hql="""
			use %(dbName)s;
			insert overwrite table %(tbName)s partition(dt='%(end_date)s')
			--1表示同设备，2表示同IP，3表示同联系人、4表示联系人是注册用户手机号、5表示联系人在注册用户通讯录中、6表示联系人在注册用户通话记录中，7表示同通讯录，8表示同通话记录
			select
				case when t.stat_type='device' then 1 else 2 end as type
				,t.user_id as from_id
				,s.user_id as to_id
				,count(distinct t.id) as weight
			from 
				(
					select
						stat_type
						,user_id
						,id
					from 
						dm_loan_usr_visit_relation_stat_td
					where 
						dt='%(end_date)s'
				) t 
			join 
				(
					select
						stat_type
						,user_id
						,id
					from 
						dm_loan_usr_visit_relation_stat_td
					where 
						dt='%(end_date)s'
				) s 
			on 
				t.id=s.id 
				and t.stat_type=s.stat_type 
			where 
				t.user_id<>s.user_id
			group by 
				case when t.stat_type='device' then 1 else 2 end
				,t.user_id
				,s.user_id
			union all 
			select
				3 as type 
				,t.user_id as from_id
				,s.user_id as to_id
				,count(distinct t.id) as weight
			from 
				(
					select
						user_id
						,id
					from 
						(
							select	
								user_id 
								,phone_md5 as id 
								,reverse(substr(reverse(trim(phone_encrypt)),1,11)) as tmp
							from
							    dwd.dwd_user_borrower_user_link_info_full
							where
							    dt='%(max_date)s'
							    and to_date(create_tm)<='%(end_date)s'
						) t 
					where 
						length(tmp)=11
						and tmp like '1%%'
					group by 
						user_id
						,id
				) t 
			join 
				(
					select
						user_id
						,id
					from 
						(
							select	
								user_id 
								,phone_md5 as id 
								,reverse(substr(reverse(trim(phone_encrypt)),1,11)) as tmp
							from
							    dwd.dwd_user_borrower_user_link_info_full
							where
							    dt='%(max_date)s'
							    and to_date(create_tm)<='%(end_date)s'
						) t 
					where 
						length(tmp)=11
						and tmp like '1%%'
					group by 
						user_id
						,id
				) s 
			on 
				t.id=s.id 
			where 
				t.user_id<>s.user_id
			group by 
				t.user_id
				,s.user_id
			union all 
			select
				4 as type 
				,t.user_id as from_id
				,s.user_id as to_id
				,count(distinct t.id) as weight
			from 
				(
					select
						user_id
						,id
					from 
						(
							select	
								user_id 
								,phone_md5 as id 
								,reverse(substr(reverse(trim(phone_encrypt)),1,11)) as tmp
							from
							    dwd.dwd_user_borrower_user_link_info_full
							where
							    dt='%(max_date)s'
							    and to_date(create_tm)<='%(end_date)s'
						) t 
					where 
						length(tmp)=11
						and tmp like '1%%'
					group by 
						user_id
						,id
				) t 
			join 
				(
					select
						user_id
						,login_phone_md5 as id
					from 
					    dwd.dwd_user_borrower_user_account_full
					where 
					    dt='%(max_date)s'
					    and to_date(create_tm)<='%(end_date)s'
				) s 
			on 
				t.id=s.id 
			group by 
				t.user_id
				,s.user_id
			union all
			select
				5 as type 
				,t.user_id as from_id
				,s.user_id as to_id
				,count(distinct t.id) as weight
			from 
				(

					select
						user_id
						,id
					from 
						(
							select	
								user_id 
								,phone_md5 as id 
								,reverse(substr(reverse(trim(phone_encrypt)),1,11)) as tmp
							from
							    dwd.dwd_user_borrower_user_link_info_full
							where
							    dt='%(max_date)s'
							    and to_date(create_tm)<='%(end_date)s'
						) t 
					where 
						length(tmp)=11
						and tmp like '1%%'
					group by 
						user_id
						,id
				) t 
			join 
				(
					select
						t.user_id
						,s.mobile_md5 as id 
					from 
						(
							select
								*
							from 
								dwd.dwd_user_borrower_user_account_full
							where 
								dt='%(max_date)s'
								and to_date(create_tm)<='%(end_date)s'
						) t 
					join 
						(
							select
								id_card_md5
								,mobile_md5
							from 
								(
									select
										id_card_md5
										,mobile_md5
										,reverse(substr(reverse(trim(phone_encrypt)),1,11)) as tmp
										,rank()over(partition by id_card_md5 order by query_tm desc) as rank
									from 
										dwd.dwd_user_fjr_contact_full
									where 
										dt='%(max_date)s'
										and concat(substr(query_tm,1,4),'-',substr(query_tm,5,2),'-',substr(query_tm,7,2))<='%(end_date)s'
								) t 
							where 
								rank=1
								and length(tmp)=11
								and tmp like '1%%'
							group by 
								id_card_md5
								,mobile_md5
						) s 
					on 
						t.id_card_md5=s.id_card_md5
				) s 
			on 
				t.id=s.id 
			group by 
				t.user_id
				,s.user_id 
			union all 
			select
				6 as type 
				,t.user_id as from_id
				,s.user_id as to_id
				,count(distinct t.id) as weight
			from 
				(
					select
						user_id
						,id
					from 
						(
							select	
								user_id 
								,phone_md5 as id 
								,reverse(substr(reverse(trim(phone_encrypt)),1,11)) as tmp
							from
							    dwd.dwd_user_borrower_user_link_info_full
							where
							    dt='%(max_date)s'
							    and to_date(create_tm)<='%(end_date)s'
						) t 
					where 
						length(tmp)=11
						and tmp like '1%%'
					group by 
						user_id
						,id
				) t 
			join 
				(
					select
						user_id
						,edw_md5(other_phone) as id
					from 
						mining.dm_loan_usr_mobile_call_detail_stat_td
					where 
						dt='%(end_date)s'
						and max_call_duration>=60
						and length(reverse(substr(reverse(trim(other_phone)),1,11)))=11
						and reverse(substr(reverse(trim(other_phone)),1,11)) like '1%%'
					group by 
						user_id
						,other_phone
				) s
			on 
				t.id=s.id
			group by 
				t.user_id
				,s.user_id 
			union all 
			select
				7 as type 
				,t.user_id as from_id
				,s.user_id as to_id
				,count(distinct t.id) as weight
			from 
				(
					select
						t.user_id
						,s.mobile_md5 as id 
					from 
						(
							select
								*
							from 
								dwd.dwd_user_borrower_user_account_full
							where 
								dt='%(max_date)s'
								and to_date(create_tm)<='%(end_date)s'
						) t 
					join 
						(
							select
								id_card_md5
								,mobile_md5
							from 
								(
									select
										id_card_md5
										,mobile_md5
										,reverse(substr(reverse(trim(phone_encrypt)),1,11)) as tmp
										,rank()over(partition by id_card_md5 order by query_tm desc) as rank
									from 
										dwd.dwd_user_fjr_contact_full
									where 
										dt='%(max_date)s'
										and concat(substr(query_tm,1,4),'-',substr(query_tm,5,2),'-',substr(query_tm,7,2))<='%(end_date)s'
								) t 
							where 
								rank=1
								and length(tmp)=11
								and tmp like '1%%'
							group by 
								id_card_md5
								,mobile_md5
						) s 
					on 
						t.id_card_md5=s.id_card_md5
				) t 
			join 
				(
					select
						t.user_id
						,s.mobile_md5 as id 
					from 
						(
							select
								*
							from 
								dwd.dwd_user_borrower_user_account_full
							where 
								dt='%(max_date)s'
								and to_date(create_tm)<='%(end_date)s'
						) t 
					join 
						(
							select
								id_card_md5
								,mobile_md5
							from 
								(
									select
										id_card_md5
										,mobile_md5
										,reverse(substr(reverse(trim(phone_encrypt)),1,11)) as tmp
										,rank()over(partition by id_card_md5 order by query_tm desc) as rank
									from 
										dwd.dwd_user_fjr_contact_full
									where 
										dt='%(max_date)s'
										and concat(substr(query_tm,1,4),'-',substr(query_tm,5,2),'-',substr(query_tm,7,2))<='%(end_date)s'
								) t 
							where 
								rank=1
								and length(tmp)=11
								and tmp like '1%%'
							group by 
								id_card_md5
								,mobile_md5
						) s 
					on 
						t.id_card_md5=s.id_card_md5
				) s 
			on 
				t.id=s.id 
			where 
				t.user_id<>s.user_id
			group by 
				t.user_id
				,s.user_id 
			union all 
			select
				8 as type 
				,t.user_id as from_id
				,s.user_id as to_id
				,count(distinct t.id) as weight
			from 
				(
					select
						user_id
						,other_phone as id
					from 
						mining.dm_loan_usr_mobile_call_detail_stat_td
					where 
						dt='%(end_date)s'
						and max_call_duration>=60
						and length(reverse(substr(reverse(trim(other_phone)),1,11)))=11
						and reverse(substr(reverse(trim(other_phone)),1,11)) like '1%%'
					group by 
						user_id
						,other_phone
				) t 
			join 
				(
					select
						user_id
						,other_phone as id
					from 
						mining.dm_loan_usr_mobile_call_detail_stat_td
					where 
						dt='%(end_date)s'
						and max_call_duration>=60
						and length(reverse(substr(reverse(trim(other_phone)),1,11)))=11
						and reverse(substr(reverse(trim(other_phone)),1,11)) like '1%%'
					group by 
						user_id
						,other_phone
				) s 
			on 
				t.id=s.id
			where 
				t.user_id<>s.user_id
			group by 
				t.user_id
				,s.user_id 
		"""%{'dbName':dbName,'tbName':tbName,'end_date':end_date,'max_date':max_date}
	#print hql
	ht.exec_sql(hql)

if __name__=='__main__':
	dbName='mining'
	tbName='dm_loan_usr_relation_td'
	delimiter='\001'
	print 'dbName=%s'%dbName,'tbName=%s'%tbName,'delimiter=%s'%delimiter
	print 'begin...'
	createTable(dbName,tbName,delimiter)
	insertTable(dbName,tbName)
	print 'success...'

