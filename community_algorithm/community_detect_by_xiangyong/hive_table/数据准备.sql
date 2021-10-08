select           #首先了解，这个SQL是为了干什么
	s.type
	,count(distinct s.from_id) as cnt           #distinct 用于查询过程中去除重复值     就算多少个from_id
	,count(1) as cnt2                           #count(1)会统计表中所有的记录数
from    #指定申请单主表中特定分区，特定申请区间内的用户ID  作为t
	(
		select 
			user_id           #用户ID
		from 
			mining.dm_loan_usr_lend_detail_td        #申请单主表
		where 
			dt='2018-02-04'                        #分区
			and apply_date>='2017-10-11'
			and apply_date<='2017-12-31'
			--and is_expire=1
			--and overdue_days>1
		group by 
			user_id
	) t 
join 
	(
		select
			*
		from 
			mining.dm_loan_usr_relation_td          #关系表，中type表示： ；weight表示： ；
		where 
			dt='2017-12-31'
			and length(from_id)>5                   #length 返回字符串长度  ？？？？
			and length(to_id)>5
			--and type=7
			--and weight>=3
	) s 
on 
	t.user_id=s.from_id    #申请人是from_id；Hive中Join的关联键必须在ON ()中指定，不能在Where中指定，否则就会先做笛卡尔积，再过滤。
where 
	(s.type=3)
	or (s.type=4 and s.from_id<>s.to_id)
	or (s.type=7 and weight>=5)
	or (s.type=8 and weight>=3)
	or (s.type=1 and weight>=1)
group by 
	s.type
limit 10000
;

drop table mining.dm_loan_usr_social_tmp_d;         #需要新删除旧表
create table mining.dm_loan_usr_social_tmp_d as     #直接将select得结果存储为社会关系表
select
	s.from_id
	,s.to_id
from 
	(
		select 
			user_id
		from 
			mining.dm_loan_usr_lend_detail_td
		where 
			dt='2017-12-31'
			and apply_date>='2017-10-11'
			and apply_date<='2017-11-30'
			--and is_expire=1
			--and overdue_days>1
		group by 
			user_id
	) t 
join 
	(
		select
			*
		from 
			mining.dm_loan_usr_relation_td
		where 
			dt='2017-11-30'
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
;

select
	count(1) as cnt 
	,count(distinct from_id) as cnt2
	,count(distinct from_id,to_id) as cnt3
from 
	mining.dm_loan_usr_social_tmp_d
limit 10000
;
select
	count(1) as cnt 
	,count(distinct user_id) as cnt2 
	,count(distinct user_id,cluster) as cnt3
from 
	mining.dm_loan_usr_cluster_d
where 
	dt='2017-12-31'
;

select
	count(distinct t.user_id) as cnt 
from 
	(
		select
			*
		from 
			mining.dm_loan_usr_cluster_d
		where
			dt='2017-12-31'
	) t 
join 
	(
		select 
			user_id
		from 
			dwd.dwd_user_borrower_user_account_full
		where 
			dt='2017-12-31'
	) s 
on  
	t.user_id=s.user_id
limit 10000
;

select
	count(1) as apply_cnt
	,sum(1-is_pass) as refuse_cnt
	,sum(1-is_pass)/count(1) as refuse_rate
	,sum(is_expire) as expire_cnt
	,sum(case when overdue_days>1 then 1 else 0 end) as overdue_cnt
	,sum(case when overdue_days>1 then 1 else 0 end)/sum(is_expire) as overdue_rate
from 
	(
		select
			user_id
			,max(is_pass)      as is_pass
			,max(is_expire)    as is_expire
			,max(overdue_days) as overdue_days
		from 
			mining.dm_loan_usr_lend_detail_td
		where 
			dt='2018-01-23'
			and apply_date>='2017-10-11'
			and apply_date<='2017-12-31'
		group by 
			user_id
	) t 
limit 10000
;

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
			mining.dm_loan_usr_cluster_d
		where 
			dt='2017-01-01'
	) t 
left join 
	(
		select
			user_id
			,max(is_pass)      as is_pass
			,max(is_expire)    as is_expire
			,max(overdue_days) as overdue_days
		from 
			mining.dm_loan_usr_lend_detail_td
		where 
			dt='2018-02-04'
			and apply_date>='2017-10-11'
			and apply_date<='2017-12-31'
		group by 
			user_id
	) s 
on  
	t.user_id=s.user_id
group by 
	t.cluster
limit 10000
;

select
	count(distinct s.user_id) as cnt 
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
					mining.dm_loan_usr_cluster_d
				where 
					dt='2017-12-31'
			) t 
		left join 
			(
				select
					user_id
					,max(is_pass)      as is_pass
					,max(is_expire)    as is_expire
					,max(overdue_days) as overdue_days
				from 
					mining.dm_loan_usr_lend_detail_td
				where 
					dt='2018-01-23'
					and apply_date>='2017-10-11'
					and apply_date<='2017-12-31'
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
			mining.dm_loan_usr_cluster_d
		where 
			dt='2017-12-31'
	) s 
on 
	t.cluster=s.cluster
where 
	t.refuse_rate>=0.7
	and t.overdue_rate>=0.7
;
#hbase表创建

drop table mining.dm_loan_usr_social_tmp_d2;
create table mining.dm_loan_usr_social_tmp_d2 as 
select
	w.login_phone_md5 as mobile_md5
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
					mining.dm_loan_usr_cluster_d
				where 
					dt='2017-12-31'
			) t 
		left join 
			(
				select
					user_id
					,max(is_pass)      as is_pass
					,max(is_expire)    as is_expire
					,max(overdue_days) as overdue_days
				from 
					mining.dm_loan_usr_lend_detail_td
				where 
					dt='2018-01-23'
					and apply_date>='2017-10-11'
					and apply_date<='2017-12-31'
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
			mining.dm_loan_usr_cluster_d
		where 
			dt='2017-12-31'
	) s 
on 
	t.cluster=s.cluster
join
	(
		select 
			user_id
			,login_phone_md5
			,id_card_md5
			,to_date(create_tm) as register_date
		from 
			dwd.dwd_user_borrower_user_account_full
		where 
			dt='2018-01-24'
	) w 
on 
	s.user_id=w.user_id
where 
	t.refuse_rate>=0.7
	and t.overdue_rate>=0.7
group by 
	w.login_phone_md5
;

