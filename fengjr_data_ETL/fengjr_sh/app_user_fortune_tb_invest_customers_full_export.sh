set -e
export_db=db_wealth_sales_full_info
before_exec="delete from TB_INVEST_CUSTOMERS where 1=1;"
source_expression="select id,full_name,phone_mobile,id_card_no,nick_name,register_datetime,user_id,overseas_regsiter_datetime,overseas_user_id,intention_extent,intention_detail_desc,intention_product_type,plan_call_date_last,lock_date,bind_source,bind_sam_id,bind_group_manager_id,bind_city_manager_id,invest_amount,shall_received_payment_time,shall_received_payment_amount,last_received_payment_time,last_received_payment_amount,cast_in_invenst_amount,total_invenst_amount,total_invenst_wealth_amount,total_invenst_overseas_amount,seven_day_invenst_amount,qualified_cert,bind_sam_name,qualified_cert_lose_date from dwi.dwi_user_fortune_tb_invest_customers_full where dt='#exec_tm#';"
target=TB_INVEST_CUSTOMERS
parallel=1
data_date=${1}
db_type=mysql
target_columns="id,full_name,phone_mobile,id_card_no,nick_name,register_datetime,user_id,overseas_regsiter_datetime,overseas_user_id,intention_extent,intention_detail_desc,intention_product_type,plan_call_date_last,lock_date,bind_source,bind_sam_id,bind_group_manager_id,bind_city_manager_id,invest_amount,shall_received_payment_time,shall_received_payment_amount,last_received_payment_time,last_received_payment_amount,cast_in_invenst_amount,total_invenst_amount,total_invenst_wealth_amount,total_invenst_overseas_amount,seven_day_invenst_amount,qualified_cert,bind_sam_name,qualified_cert_lose_date"
exp_switch=${2}

cd ../../../sbin/
sh job_core_export.sh "${export_db}" "${before_exec}" "${source_expression}" "${target}" "${parallel}" "${data_date}" "${db_type}" "${target_columns}" "${exp_switch}"