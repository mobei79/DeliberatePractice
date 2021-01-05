# -*- coding: utf-8 -*-
import pickle
import json
from logger import Logger
import pandas as pd
from sklearn import preprocessing

logger = Logger(logname='log.txt', loglevel=1, logger="fox").getlog()


class Magpie_Pass_Tree():
    def __init__(self):
        self.model = None
        self.features =[ 'fjthxd_111_v_net_in_days', 'fjthxd_111_v_bill_charge_td', 'fjthxd_111_v_bill_charge_avg_month', 'fjthxd_111_v_tel_package_fee_avg', 'fjthxd_111_v_bill_charge_last_month', 'fjthxd_111_v_tel_package_fee_last_month', 'fjthxd_111_v_net_flow_num_td', 'fjthxd_111_v_net_flow_num_30d', 'fjthxd_111_v_net_flow_num_w', 'fjthxd_111_v_sms_num_td', 'fjthxd_111_v_sms_num_30d', 'fjthxd_111_v_sms_num_w', 'fjthxd_111_v_send_sms_num_td', 'fjthxd_111_v_send_sms_num_30d', 'fjthxd_111_v_send_sms_num_w', 'fjthxd_111_v_rec_sms_num_td', 'fjthxd_111_v_rec_sms_num_30d', 'fjthxd_111_v_rec_sms_num_w', 'call_cnt_7d', 'call_usr_num_7d', 'call_month_num_7d', 'call_day_num_7d', 'call_duration_7d', 'max_call_duration_7d', 'min_call_duration_7d', 'avg_call_duration_7d', 'call_out_cnt_7d', 'call_out_usr_num_7d', 'call_out_month_num_7d', 'call_out_day_num_7d', 'call_out_duration_7d', 'max_call_out_duration_7d', 'min_call_out_duration_7d', 'avg_call_out_duration_7d', 'call_in_cnt_7d', 'call_in_usr_num_7d', 'call_in_month_num_7d', 'call_in_day_num_7d', 'call_in_duration_7d', 'max_call_in_duration_7d', 'min_call_in_duration_7d', 'avg_call_in_duration_7d', 'call_cnt_15d', 'call_usr_num_15d', 'call_month_num_15d', 'call_day_num_15d', 'call_duration_15d', 'max_call_duration_15d', 'min_call_duration_15d', 'avg_call_duration_15d', 'call_out_cnt_15d', 'call_out_usr_num_15d', 'call_out_month_num_15d', 'call_out_day_num_15d', 'call_out_duration_15d', 'max_call_out_duration_15d', 'min_call_out_duration_15d', 'avg_call_out_duration_15d', 'call_in_cnt_15d', 'call_in_usr_num_15d', 'call_in_month_num_15d', 'call_in_day_num_15d', 'call_in_duration_15d', 'max_call_in_duration_15d', 'min_call_in_duration_15d', 'avg_call_in_duration_15d', 'call_cnt_30d', 'call_usr_num_30d', 'call_month_num_30d', 'call_day_num_30d', 'call_duration_30d', 'max_call_duration_30d', 'min_call_duration_30d', 'avg_call_duration_30d', 'call_out_cnt_30d', 'call_out_usr_num_30d', 'call_out_month_num_30d', 'call_out_day_num_30d', 'call_out_duration_30d', 'max_call_out_duration_30d', 'min_call_out_duration_30d', 'avg_call_out_duration_30d', 'call_in_cnt_30d', 'call_in_usr_num_30d', 'call_in_month_num_30d', 'call_in_day_num_30d', 'call_in_duration_30d', 'max_call_in_duration_30d', 'min_call_in_duration_30d', 'avg_call_in_duration_30d', 'call_cnt_3m', 'call_usr_num_3m', 'call_month_num_3m', 'call_day_num_3m', 'call_duration_3m', 'max_call_duration_3m', 'min_call_duration_3m', 'avg_call_duration_3m', 'call_out_cnt_3m', 'call_out_usr_num_3m', 'call_out_month_num_3m', 'call_out_day_num_3m', 'call_out_duration_3m', 'max_call_out_duration_3m', 'min_call_out_duration_3m', 'avg_call_out_duration_3m', 'call_in_cnt_3m', 'call_in_usr_num_3m', 'call_in_month_num_3m', 'call_in_day_num_3m', 'call_in_duration_3m', 'max_call_in_duration_3m', 'min_call_in_duration_3m', 'avg_call_in_duration_3m', 'call_cnt_td', 'call_usr_num_td', 'call_month_num_td', 'call_day_num_td', 'call_duration_td', 'max_call_duration_td', 'min_call_duration_td', 'avg_call_duration_td', 'call_out_cnt_td', 'call_out_usr_num_td', 'call_out_month_num_td', 'call_out_day_num_td', 'call_out_duration_td', 'max_call_out_duration_td', 'min_call_out_duration_td', 'avg_call_out_duration_td', 'call_in_cnt_td', 'call_in_usr_num_td', 'call_in_month_num_td', 'call_in_day_num_td', 'call_in_duration_td', 'max_call_in_duration_td', 'min_call_in_duration_td', 'avg_call_in_duration_td']

    def get(self):
        with open('test_model.pkl', 'rb') as f:
            self.model = pickle.loads(f.read())
        return self 

    def predict(self, input_json):
        # 帅选字段
        pre_features = json.loads(input_json, encoding='utf-8')
        pre_features = pre_features['data']
        df = pd.DataFrame.from_dict(pre_features, orient='index').T
        df = df.fillna(0)
        data = df[self.features]
        # 模型预测
        score = self.model.predict_proba(data)[0][1]
        result = int(score > 0.4)
        # 结果返回
        return json.dumps(dict(score=score, result=result, features=self.features))

if __name__ == "__main__":
    showbilr = Magpie_Pass_Tree()
    showbilr.get()
    data1 = pd.read_excel("query_result-79.xlsx")
    data = pd.read_table("test_model_data_1.txt", sep="\t")

    new_list = []
    for element in data1.columns:
        new_list.append(element.split(".")[1])
    data.columns = new_list

    del data["apply_id"]
    del data["apply_time"]
    del data["channel"]
    del data["user_id"]
    del data["id_card_md5"]
    del data["dt"]
    data = data.fillna(0)
    data = data.iloc[:, 1:]
    X_columns = data.columns
    for index, row in data.iterrows():
        input = json.dumps(dict(data=dict(row)))
        print(showbilr.predict(input))


