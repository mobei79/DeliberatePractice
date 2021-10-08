insert overwrite table %s partition (dt='%s')
select
    loan_id,
    user_id,
    periods_no,       --期数
    due_total_amount, --逾期总金额
    due_time,       --逾期时间
    penalty_date,     -- 罚金时间
    fact_total_amount,  --实际总金额
    plan_start_time,  -- 计划开始时间
    fact_time,      -- 实际开始时间
    repay_status,   --还款状态
    plan_status   --计划状态
from
    ods.biz_loan_core_assets_repay_plan_full
where
    dt='%s'
    and due_time='%s'
    and (plan_status='CLEARED' or plan_status='OVERDUE')
    and periods_no != 0
