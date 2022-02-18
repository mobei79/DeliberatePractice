-- 逾期预测
-- 逾期预测1 - sample_extract 从还款记录表抽取目标样本用户，如我们的模型是预测用户是否会逾期？使用七天后到期的用户，是为了可以进行迭代验证，使用基本信息，以及15天内行为距离；
show create table ods.biz_loan_core_assets_repay_plan_full -- 商业贷款核心资产全额偿还计划
	-- biz商业_loan借贷_core核心_assets资产_repayplan还款计划_full全表
	-- 规则 抽取还款计划表中七天后到期，且未还款的用户作为训练用户
insert overwrite default.dm_user_overdue_predict_sample partition (dt='2021-09-18')
-- 每步数据都要落表一次，保存数据，今天（18）抽取的是昨天（17）的数据
-- 此处是以标的为主键，一个申请是一个标的，一个人可以有多个申请，多个loan_id，但是同时只有一期到期，超过了就算逾期了
select
	loan_id,
    user_id,
    periods_no,
    due_total_amount, -- 本期应还总额
    due_time,  -- 本期应还时间
    penalty_date,  -- 罚息计算日期
    fact_total_amount,  -- 本期实际还款金额
    plan_start_time,  --  还款计划开始生效时间
    fact_time,  -- 本期实际还款时间
    repay_status,  -- 支付还款状态
    plan_status  -- 还款计划状态
from
    ods.biz_loan_core_assets_repay_plan_full
where
    dt = '2021-09-17'
    and due_time = '2021-09-24'  -- = cur_data-datetime.timedelta(day=7) 七天后到期
    and plan_status = 'UNDUE'
	--* 未到期 UNDUE("待还款"),PAYING("支付中")
	--* 所有还款成功结束 PRE_CLEARED("提前结清"),CLEARED("已还清"),OVERDUE_CLEARED("逾期结清")
	-- * 逾期未归还，任何一期还款超过dueDate都自动转为此状态  OVERDUE("已逾期");
    and fact_time is null -- 本期实际还款时间
    and periods_no != 0

-- 逾期预测 - feature_extract 抽取目标样本用户的基础数据，如 逾期利率、贷款利率、账户金额、贷款模式、职业、收入、信用卡信息余额、城市、信用评分、
-- 公司性质、信用等级，债务比例、担保信息、是否逾期等
create table default.dm_user_overdue_predict_feature_tmp2 as  -- 临时存储初版特征
select
	a.*,
	b.advance_rate,
    b.caution_amount,
    b.caution_status,
    b.guarantee_id,
    b.interest_start_date,
    b.loan_rate,
    b.loan_service_rate,
    b.max_overdue_amount,
    b.on_account_amount,
    b.overdue_rate,
    b.repayment_periods_no,
    b.source_channel,
    b.update_tm,
	c.asset_state,
    c.borrow_money,
    c.borrow_month,
    c.borrow_purpose,
    c.borrow_purpose_desc,
    c.borrow_rate,
    c.business_model,
    c.car_loan,
    c.census_register,
    c.channel,
    c.company_business,
    c.company_nature,
    c.company_size,
    c.cooperation_model,
    c.credit_grade,
    c.credit_status,
    c.debt_ratio,
    c.download_times,
    c.education,
    c.guarantee_organization,
    c.guarantee_way,
    c.guaranty_info,
    c.house_loan,
    c.income,
    c.income_truth,
    c.is_car,
    c.is_children,
    c.is_house,
    c.is_married,
    c.is_policy,
    c.is_socialsecurity,
    c.opening_bank,
    c.org_id,
    c.position,
    c.product_name,
    c.product_rate,
    c.product_type_id,
    c.profession_status,
    c.repayment_way,
    c.risk_rank,
    c.settled_tm,
    c.sex,
    c.state_change_tm,
    c.verified_income,
    c.withdraw_status,
    c.work_city,
    c.work_province,
    c.working_years,
    c.working_years_truth,
    c.contract_no,
	d.ad_code,
    d.card_auth_status,
    d.credit_card_amount,
    d.credit_card_status,
    d.mobile_city,
    d.mobile_country,
    d.operator_auth_status,
    d.register_source_code,
    d.zmxy_auth_status,
    e.age,
    f.ac_record,
    f.access_type,
    f.address,
    f.apply_amount,
    f.appnum,
    f.bank_name,
    f.bh_url_flag,
    f.birth_date,
    f.bus_type,
    f.bussiness,
    f.career_status,
    f.channel_request_tm,
    f.company_address,
    f.company_city,
    f.company_name,
    f.company_province,
    f.company_town,
    f.credit_card_bank_name,
    f.device_type,
    f.face_score,
    f.firstcategory,
    f.gender,
    f.gps_area,
    f.gps_city,
    f.gps_province,
    f.has_contacts,
    f.housing_fund_status,
    f.income_tm,
    f.is_dial_confirm,
    f.is_dial_type,
    f.is_dial_typehuman,
    f.is_dial_typem,
    f.is_helical_accelerator,
    f.is_root,
    f.is_trans,
    f.is_virtualmachine,
    f.jail_break,
    f.job_salary,
    f.label_id,
    f.label_name,
    f.latitude,
    f.length_of_residence_month,
    f.length_of_residence_year,
    f.loan_amount,
    f.loan_use,
    f.longitude,
    f.manual_check,
    f.max_acceptable_monthly_payment,
    f.mobile_brand,
    f.mobile_os,
    f.mobile_type,
    f.monthly_repay_amount,
    f.nation,
    f.networktype,
    f.occupation,
    f.pboc_credit_status,
    f.period,
    f.pre_grant_credit_amount,
    f.pre_grant_credit_term,
    f.product_type_name,
    f.profession,
    f.profession_code,
    f.profession_station,
    f.register_channel,
    f.register_channel_name,
    f.resident_city,
    f.resident_province,
    f.resident_province2,
    f.resident_town,
    f.risk_lead_flag,
    f.scene_id,
    f.sign_issue_org,
    f.simulator,
    f.start_work_tm,
    f.status,
    f.system_tag,
    f.total_repay_amount,
    f.user_level,
    f.user_risk_score,
    f.white_list_flag,
    f.white_list_level,
    f.white_list_type,
    f.work_position,
    f.work_tm,
    f.xhd_white_list_flag,
    f.zmxy_auth_expiry_tm,
    f.zmxy_status,
    g.ad_name,
    g.asset_create_tm,
    g.asset_result_version,
    g.cajl_risk_result_state,
    g.cajl_risk_result_state_desc,
    g.channel_code,
    g.channel_first_cate_code,
    g.channel_first_cate_name,
    g.channel_name,
    g.channel_second_cate_code,
    g.channel_second_cate_name,
    g.credit_end_tm,
    g.credit_state,
    g.credit_state_desc,
    g.end_result,
    g.end_result_desc,
    g.grant_credit_result,
    g.grant_credit_result_desc,
    g.info_check_result,
    g.info_check_result_desc,
    g.info_check_user,
    g.manu_info_check_result,
    g.manu_info_check_result_desc,
    g.manucheck_advice,
    g.manucheck_risk_level,
    g.manucheck_state,
    g.manucheck_state_desc,
    g.manucheck_user,
    g.retry_times,
    g.risk_result_state,
    g.risk_result_state_desc,
    g.system_risk_level,
    h.his_overdue_cnt,
    h.max_his_overdue_days,
    h.max_his_overdue_aging,
    h.max_his_overdue_days_12m,
    h.max_his_overdue_aging_12m,
    h.his_overdue_cnt_6m,
    h.max_his_overdue_days_6m,
    h.max_his_overdue_aging_6m,
    h.his_overdue_amt,
    h.his_overdue_cnt_12m,
    h.his_overdue_amt_12m,
    h.his_overdue_amt_6m,
    h.his_overdue_d3_cnt,
    h.his_overdue_d3_cnt_12m,
    h.his_overdue_d3_amt,
    h.his_overdue_d3_cnt_6m,
    h.his_overdue_d3_amt_12m,
    h.his_overdue_d3_amt_6m,
    h.his_overdue_d7_cnt,
    h.his_overdue_d7_amt,
    h.his_overdue_d15_cnt,
    h.his_overdue_d15_cnt_12m,
    h.his_overdue_d15_cnt_6m,
    h.his_overdue_d15_amt,
    h.his_overdue_d15_amt_12m,
    h.his_overdue_d15_amt_6m,
    h.overdue_settle_cnt,
    h.cur_aging,
    h.cur_overdue_days,
    h.cur_overdue_amt,
    h.overdue_settle_amt,
    h.his_overdue_m1_cnt,
    h.his_overdue_m1_cnt_12m,
    h.his_overdue_m1_cnt_6m,
    h.his_overdue_m1_amt,
    h.his_overdue_m1_amt_12m,
    h.his_overdue_m1_amt_6m,
    h.his_overdue_m2_cnt,
    h.his_overdue_m2_cnt_12m,
    h.his_overdue_m2_cnt_6m,
    h.his_overdue_m2_amt,
    h.his_overdue_m2_amt_12m,
    h.his_overdue_m2_amt_6m,
    h.his_overdue_m3_cnt,
    h.his_overdue_m3_cnt_12m,
    h.his_overdue_m3_amt,
    h.his_overdue_m3_amt_12m,
    h.his_overdue_m3_cnt_6m,
    h.his_overdue_m3_amt_6m,
    h.his_overdue_m6_cnt,
    h.his_overdue_m6_amt,
    h.his_overdue_m6_cnt_12m,
    h.his_overdue_m6_amt_12m,
    i.total_periods_no,
    i.due_limit,
    i.due_limit_unit,
    i.repayment_type,
    i.risk_score,
    i.organization
from
	(select * from default.dm_user_overdue_predict_sample where dt = '2021-09-18') a -- sample是上步生成的，是当前时间
	left join (select * from dwd.dwd_ordr_magpie_biz_assets_loan_full) b on (a.loan_id = b.loan_id)  -- 喜鹊快贷-借款还款主表
	left join (select * from dwd.dwd_ordr_p2p_asset_info_full where dt = '%s') c on (a.loan_id = c.loan_id) --'网贷:资产端-资产信息表
	left join (select * from dim.dim_user_magpie_user_account_full) d on (a.user_id = d.user_id)  -- 喜鹊快贷用户表
    left join (select * from dim.dim_user_magpie_user_loan_receipt_full) e on (a.user_id = e.user_id)  -- 喜鹊快贷进件用户表'
    left join (select * from dwd.dwd_ordr_magpie_risk_asset_info_full where dt = '%s') f on (c.contract_no=f.real_income_no)  -- 资产信息表
    left join (select * from dwi.dwi_ordr_magpie_risk_asset_risk_result_full where dt = '%s') g on (f.id=g.asset_id);  -- 渠道资产风控结果
	left join (select * from dwi.dwi_ordr_p2p_asset_repay_info_full where dt = '%s') h on (a.loan_id=h.loan_id)  -- 贷后-还款表
    left join (select * from ods.biz_loan_platform_loan_receipt_full where dt='%s') i on (a.loan_id=i.loan_id)  --网贷进件信息总表
	-- left join左表为主表，以左表字段为准，如果右边没有则为空；

-- 简化 因为每个贷款申请是一个标，进到fengjrAPP中生成标的loan_id，然后在募集资金形成对应关系；一个人可能有多个标的；
-- 所以这里使用 row_number()over(partition by loan_id order by age desc) 形成一个排序值；每个标的只取num=1的；
insert overwrite table default.dm_user_overdue_predict_feature_tmp
select * from
(
    select *, row_number() over (partition by loan_id order by age desc) num from dm_user_overdue_predict_feature_tmp2
) t where t.num=1;
drop table dm_user_overdue_predict_feature_tmp2





-- 逾期预测 -- 抽取用户行为 原版数据 抽取近7天的行为日志
insert overwrite table dm_user_overdue_user_app_log_tmp
select
	b.*
from
	(select * from default.dm_user_overdue_predict_sample  where dt = '2021-09-18') a
	join
	(select * from dwd.dwd_flow_lending_app_log where dt<='2021-09-18' and dt>='2021-09-11') b
	on a.user_id=b.user_id

-- 行为数据主要是网页埋点得到的数据，
-- 主要看行为埋点表中：ct_page：（构造app log特征所需的app页面名称）；click_name：（构造app log特征所需的点击行为名称）
-- 然后按照用户统计他们在表中的行为数目，如触发次数等等信息，统计如：是否第一次访问，是否最后访问，15天访问次数，15天点击次数，15天视图次数等等
-- 计算这些行为数据的时候 需要特别注意 按照特征名称命名规范；厂商_hive表号_特征类型_特征含义名(_时间维度), 样例："jxl_13_t_user_city"


# raw_feature_2021-09-22_2021-09-27.csv 未加工样本共（264 + 392）656个，包含loan_id,user_id等非特征信息；
# feature_process_pre.py 特征工程
按照periods_no和his_overdue_cnt分为：首期还款逾期预测、无历史逾期用户逾期预测、有历史逾期用户逾期预测三类
1. 样本分类:分为首逾，首期，已逾
2. 样本筛选1 删除指定范围内的特征，如删除期数为0，借贷金额小于多少的，地区不是什么范围内的； 使用dataform[dataform[feature] <=> range] 来实现
3. 特征删选 需要在config配置中写入已经选定的特征：
    首逾选取106个特征，历史逾期选择43个特征；
4. 特征保存保存中间数据。删除 'user_id', 'loan_id'
5. 特征处理 缺失值填充
6. 特征分箱
    读取config分箱边界；
    根据特征类型和特征边界进行分箱；
7. 读取woe tabel；根据woe tabel进行woe特征编码；【woe编码是，tabel是训练时确定的，每种类型都有一个woe值，使用时映射过去即可】
    保存woe值
