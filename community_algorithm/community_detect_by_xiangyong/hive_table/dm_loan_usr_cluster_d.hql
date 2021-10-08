use mining;
create table if not exists dm_loan_usr_cluster_d
(
    user_id		 	   	string
    ,cluster            int
)
partitioned by (dt string comment 'dt')
row format delimited
fields terminated by '%(delimiter)s'
stored as textfile