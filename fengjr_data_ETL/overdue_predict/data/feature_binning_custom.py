# -*- coding: utf-8 -*-
# !/usr/bin/env python
import math




#####################################################
# loan_asset的信息


def get_periods_no(x):
    array = [0]*16
    try:
        value = int(x)
        if value >= 0 and value <= 14:
            array[value] = 1
        else:
            array[15] = 1
    except:
        print("get_periods_no") 
        array[15] = 1
    
    return array

def get_total_periods_no(x):
    try:
        x = int(x)
    except:
        x = 0
    cases = [0,3,6,9,12,24 ,36]
    if x in cases:
        idx = cases.index(x)
    else:
        idx = 0
    array = [0]*len(cases)
    array[idx] = 1
    return array


def get_due_limit(x):
    idx_map = {"3MONTH":0, "6MONTH":1, "9MONTH":2, "12MONTH":3, "24MONTH":4, "36MONTH":5}
    idx = idx_map.get(x,5)
    array = [0] * 6
    array[idx] = 1
    return array
    

def get_risk_score(x):
    ### 将空的放到默认档
    if x == "" or x == "NULL":
        x = 4000
    try:
        x = float(x)
    except:
        x = 4000
    
    if x < 462.24:
        idx = 0
    elif x < 591.0:
        idx = 1
    elif x < 614.0:
        idx = 2
    elif x < 756:
        idx = 3
    elif x < 3600:
        idx = 4
    else:
        idx = 5
    array =[0]*6
    array[idx] = 1
    return array
    
def get_loan_amount(x):
    try:
        x = float(x)
    except:
        x = 20000
    if x < 5000:
        idx = 0
    elif x < 10896:
        idx = 1
    elif x < 11947:
        idx = 2
    elif x < 12706:
        idx = 3
    elif x < 16000:
        idx = 4
    else:
        idx = 5
    array = [0] * 6
    array[idx] = 1
    return array

def get_repayment_periods_no(x):
    try:
        x = int(x)
    except:
        x = 0

    if x == 0:
        idx = 5
    elif x <= 1:
        idx = 0
    elif x <= 2:
        idx = 1
    elif x <= 3:
        idx = 2
    elif x <= 6:
        idx = 3
    else:
        idx = 4
    array = [0] * 6
    array[idx] = 1
    return array
    

def get_advance_rate(x):
    try:
        x = int(x)
    except:
        x = -1

    if x == 0:
        idx = 0
    elif x == 1500:
        idx = 1
    else:
        idx = 2
    array = [0]*3
    array[idx] = 1
    return array


def get_max_overdue_amount(x):
    try: 
        x = float(x)
    except:
        x = -1
    if x <= 0 :
        idx = 6
    elif x < 5000:
        idx = 0
    elif x < 10896:
        idx = 1
    elif x < 11947:
        idx = 2
    elif x < 12706:
        idx = 3
    elif x < 16000:
        idx = 4
    else:
        idx = 5
    array = [0] * 7
    array[idx] = 1
    return array

def get_age(x):
    try:
        x = int(x)
    except:
        x = -1
    if x <= 0:
        idx = 7
    elif x <= 18:
        idx = 0
    elif x <= 27:
        idx = 1
    elif x <= 31:
        idx = 2
    elif x <= 34:
        idx = 3
    elif x <= 40:
        idx = 4
    elif x <= 56:
        idx = 5
    else:
        idx = 6
    array = [0]*8
    array[idx] = 1
    return array


def get_is_married(x):
    if x == "未婚":
        idx = 0
    elif x == "已婚":
        idx = 1
    elif x == "离异":
        idx = 2
    elif x == "丧偶":
        idx = 3
    else:
        idx = 4
    array = [0]*5
    array[idx] = 1
    return array


def get_education(x):
    if x == "中专":
        idx = 0
    elif x == "大专":
        idx = 1
    elif x == "高中及以下":
        idx = 2
    elif x == "硕士":
        idx = 3
    elif x == "本科":
        idx = 4
    elif x == "博士":
        idx = 5
    else:
        idx = 6
    array = [0]*7
    array[idx] = 1
    return array


def get_borrow_rate(x):
    try:
        x = float(x)
    except:
        x = -1
    if x < 0:
        idx = 7
    elif x <= 7.0:
        idx = 0
    elif x <= 7.8:
        idx = 1
    elif x <= 8:
        idx = 2
    elif x <= 8.5:
        idx = 3
    elif x <= 8.8:
        idx = 4
    elif x <= 9:
        idx = 5
    elif x <= 9.3:
        idx = 6
    else:
        idx = 7
    array = [0]*8
    array[idx] = 1
    return array

def get_guarantee_organization(x):
    try:
        x = int(x)
    except:
        x = -1
    if x == 53:
        idx = 0
    elif x == 39:
        idx = 1
    elif x == 14:
        idx = 2
    else:
        idx = 3
    array = [0]*4
    array[idx] = 1
    return array

def get_income(x):
    if x == "5001-10000":
        idx = 0
    elif x == "2001-5000":
        idx = 1
    elif x == "10001-20000":
        idx = 2
    elif x == "20001-50000":
        idx = 3
    elif x == "50001以上":
        idx = 4
    elif x == "2000以下":
        idx = 5
    else:
        idx = 6
    array = [0]*7
    array[idx] = 1
    return array


def get_company_business(x):
    if x == "其他":
        idx = 0
    elif x == "科学研究/技术服务":
        idx = 1
    elif x == "未知":
        idx = 2
    else:
        idx = 3
    array = [0]*4
    array[idx] = 1
    return array


def get_asset_state(x):
    if x == "WITHDRAWSUCCEEDED":
        idx = 0
    elif x == "CLEARED":
        idx = 1
    elif x == "FINISHED":
        idx = 2
    elif x == "INITIATED":
        idx = 3
    elif x == "SCHEDULED":
        idx = 4
    else:
        idx = 5
    array = [0]*6
    array[idx] = 1
    return array

def get_cooperation_model(x):
    if x ==  "SINGLE_AND_REGULAR":
        idx = 0
    elif x == "SINGLE":
        idx = 1
    elif x == "REGULAR":
        idx = 2
    else:
        idx =  3
    array =[0]*4
    array[idx] = 1
    return array 


def get_verified_income(x):
    try:
        x = float(x)
    except:
        x = -1
    if x < 0:
        idx = 6
    if x>=1000000:
        idx = 7
    elif x < 6000:
        idx = 0
    elif x < 8000:
        idx = 1
    elif x < 10000:
        idx = 2
    elif x < 17500:
        idx = 3
    elif x <30000:
        idx = 4
    else:
        idx = 5
    array = [0]*8
    array[idx] = 1
    return array


def get_channel(x):
    if x == "NULL":
        idx = 0
    elif x == "2":
        idx = 1
    elif x == "9":
        idx = 2
    elif x == "4":
        idx = 3
    elif x == "5":
        idx = 4
    elif x == "10":
        idx = 5
    else:
        idx = 6
    array = [0]*7
    array[idx] = 1
    return array

def get_margin_ratio(x):
    try:
        x = float(x)
    except:
        x = 22 ### 为了落入默认分段中
    if x < 6.0:
        idx = 0
    elif x <= 7.5:
        idx = 1
    elif x <= 14.0:
        idx = 2
    elif x <= 16.0:
        idx = 3
    elif x <= 21.0:
        idx = 4
    else:
        idx = 5
    array = [0]*6
    array[idx] = 1
    return array

def get_caution_amount(x):
    try:
        x = float(x)
    except:
        x = 0.0  ###落入默认的分段当中
    if x < 1e-6:
        idx = 0
    elif x < 1766.45:
        idx = 1
    else:
        idx = 2
    array = [0]*3
    array[idx] = 1
    return array


def get_loan_rate(x):
    try:
        x = float(x)
    except:
        x = -1  ### 落入默认分段
    if x < 1e-5:
        idx = 0
    elif x <= 850:
        idx = 1
    elif x <= 900:
        idx = 2
    else:
        idx = 3
    array = [0]*4
    array[idx] = 1
    return array

def get_withdraw_status(x):
    if x == "WITHDRAW_SUCCESS":
        idx = 0
    else:
        idx = 1
    array = [0]*2
    array[idx] = 1
    return array

def get_sex(x):
    if x == "男":
        idx = 0
    elif x == "女":
        idx = 1
    else:
        idx = 2
    array = [0]*3
    array[idx] = 1
    return array


def get_borrow_money(x):
    try:
        x = float(x)
    except:
        x = -1

    if x < 0:
        idx = 0
    elif x < 9685:
        idx = 1
    elif x < 11947:
        idx = 2
    elif x < 12706:
        idx = 3
    elif x < 16000:
        idx = 4
    else:
        idx = 5
    array = [0]*6
    array[idx] = 1
    return array


def get_is_socialsecurity(x):
    if x == "" or x is None :
        idx = 0
    else:
        idx = 1
    array = [0]*2
    array[idx] = 1
    return array

def get_is_policy(x):
    if x == "" or x is None:
        idx = 0
    else:
        idx = 1
    array = [0]*2
    array[idx] = 1
    return array

def get_product_type_id(x):
    if x == "" or x is None:
        idx = 0
    elif x == "7":
        idx = 1
    elif x == "10":
        idx = 2
    elif x == "11":
        idx = 3
    elif x == "15":
        idx = 4
    else:
        idx = 5
    array = [0]*6
    array[idx] = 1
    return array


def get_business_model(x):
    if x == "":
        idx = 0
    elif x == "A5":
        idx = 1
    elif x == "A6":
        idx = 2
    else:
        idx = 3
    array = [0]*4
    array[idx] = 1
    return array

def get_due_total_amount(x):
    try:
        x = float(x)
    except:
        x = 1600   ### 落入默认的分区之内
    if x < 1128.41:
        idx = 0
    elif x < 1586:
        idx = 1
    else:
        idx = 2
    array = [0]*3
    array[idx] = 1
    return array


###############################################################
#risk_info的特征抽取
"""
def get_due_limit(x):
    idx_map = {"3":0, "6":1, "9":2, "12":3, "24":4, "36":5}
    idx = idx_map.get(x,5)
    array = [0] * 6
    array[idx] = 1
    return array
"""

def get_nation(x):
    if x == "汉":
        idx = 0
    else:
        idx = 1
    array = [0]*2
    array[idx] = 1
    return array


### 前面有过education的字段
def get_highest_eduction(x):
    if x == "" or x is None:
        idx = 0
    elif x == "大专":
        idx = 1
    elif x == "本科":
        idx = 2
    elif x == "高中":
        idx = 3
    elif x == "中专":
        idx = 4
    elif x == "初中":
        idx = 5
    elif x == "硕士":
        idx = 6
    elif x == "其他":
        idx = 7
    elif x == "博士":
        idx = 8
    else:
        idx = 0
    array = [0]*9
    array[idx] = 1
    return array

"""
def get_is_married(x):
    if x == "" or x is None:
        idx = 0
    elif x == "已婚":
        idx = 1
    elif x == "未婚":
        idx = 2
    elif x == "离异":
        idx = 3
    else:
        idx = 4
    array = [0]*5
    array[idx] = 1
    return array
"""

def get_profession(x):
    if x == "" or x is None:
        idx = 0
    else:
        idx = 1
    array = [0]*2
    array[idx] = 1
    return array

def get_job_salary(x):
    try:
        x = float(x)
    except:
        x = 1e8
    if x > 0 and x < 2000:
        idx = 0
    elif x < 5000:
        idx = 1
    elif x < 10000:
        idx = 2
    elif x < 20000:
        idx = 3
    elif x < 50000:
        idx = 4
    elif x >= 50000 and x < 1e8:
        idx = 5
    else:
        idx = 6
    # print(x,idx)
    array = [0]*7
    array[idx] = 1
    return array

def get_loan_use(x):
    if x == "" or x is None:
        idx  = 0
    cases = ["", "装修", "教育", "旅游", "百货消费", "创业","扩大经营", "婚庆", "其他", "购车","购房", "租房"]
    if x in cases:
        idx = cases.index(x)
    else:
        idx = len(cases)
    array = [0] * ( len(cases) + 1 )
    array[idx] = 1
    return array

def get_channel_code(x):
    if x == "" or x is None:
        idx = 0
    cases = ["", "JIANBING","APP","RONG","BEIDOU","KANIU","RONG360","YQG","JQNS"]
    if x in cases:
        idx = cases.index(x)
    else:
        idx = len(cases)
    array = [0] * ( len(cases) + 1 )
    array[idx] = 1
    return array


def get_period(x):
    cases = ["0","3","6","9","12","24","36"]
    if x in cases:
        idx = cases.index(x)
    else:
        idx = len(cases)
    array = [0] * ( len(cases) + 1) 
    array[idx] = 1
    return array

def get_start_work_tm(x):
    # x = x[:4]  ####截取时间格式中的年份数据
    try:
        x = x[:4]  ####截取时间格式中的年份数据
        x = int(x)
    except:
        x = -1

    idx = (x - 1980) // 5
    if idx < 0:
        idx = 0
    if idx > 8:
        idx = 8
    array = [0] * 10
    array[idx] = 1
    return array


def get_white_list_flag(x):
    if x == "1":
        idx = 0
    else:
        idx = 1
    array = [0] * 2
    array[idx] = 1
    return array

def get_housing_fund_status(x):
    if x == "":
        idx = 0
    elif x == "SUCCESS":
        idx = 1
    elif x == "UNAUTH":
        idx = 2
    else:
        idx=0
    array = [0] * 3 
    array[idx] = 1
    return array

def get_operator_auth_status(x):
    if x == "":
        idx = 0
    elif x == "AUTH":
        idx = 1
    elif x == "NOAUTH" or x=="UNAUTH":
        idx = 2
    else:
        idx=0
    array = [0] * 3
    array[idx] = 1
    return array


def get_credit_card_status(x):
    if x == "" or x is None:
        idx = 0
    elif x == "UNAUTH":
        idx = 1
    elif x == "SUCCESS":
        idx = 2
    else:
        idx=0
    array = [0] * 3
    array[idx] = 1
    return array

def get_pboc_credit_status(x):
    if x == "":
        idx = 0
    elif x == "UNAUTH":
        idx = 1
    elif x == "AUTH":
        idx = 2
    else:
        idx=0
    array = [0] * 3
    array[idx] = 1
    return array


def get_bh_url_flag(x):
    try:
        x = int(x)
    except:
        x = ""
    print(x)
    if x == "":
        idx = 0
    elif x == 0:
        idx = 1
    elif x == 1:
        idx = 2
    else:
        idx=0
    array = [0] * 3
    array[idx] = 1
    print(array)
    return array

def get_occupation(x):
    if x == "":
        idx = 0
    if x == "AZ":
        idx = 1
    elif x == "AY":
        idx = 2
    elif x == "A4":
        idx = 3
    else:
        idx = 0
    array = [0]*4
    array[idx] = 1
    return array

def get_career_status(x):
    if x == "A9":
        idx = 1
    elif x=='A1':
        idx = 2
    elif x=='A2':
        idx = 3
    elif x=='A3':
        idx = 4
    else:
        idx = 0
    array = [0]*5
    array[idx] = 1
    return array

def get_work_position(x):
    if x == "":
        idx = 0
    elif x == "A9":
        idx = 1
    else:
        idx = 2
    array = [0]*3
    array[idx] = 1
    return array

"""
def get_work_tm(x):
    if x == "" or x is None:
        x = -1
    x = int(x)
    idx = x // 12
    if idx > 10:
        idx = 10
    if x < 0:
        idx = 11
    array = [0] * 12
    array[idx] = 1
    return array
"""

"""
def get_gender(x):
    if x == "":
        idx = 0
    elif x == "男":
        idx = 1
    elif x == "女":
        idx = 2
    array = [0]*3
    array[idx] = 1
    return array
"""

def get_channel_first_cate_code(x):
    if x == "":
        idx = 0
    elif x == "24":
        idx = 1
    elif x == "20":
        idx = 2
    elif x == "22":
        idx = 3
    else:
        idx = 4
    array = [0] * 5
    array[idx] = 1
    return array


#################################################
#
#基于用户基础信息得到的字段

def get_card_auth_status(x):
    if x == "AUTH":
       idx = 0
    elif x == "UNAUTH":
        idx = 1
    else:
        idx = 2
    array = [0]*3
    array[idx] = 1
    return array

def get_credit_card_amount(x):
    try:
        x = float(x)
    except:
        x = -1

    if 0 < x  and x < 20000:
        idx  = 1
    else:
        idx = 2
    array = [0]*3
    array[idx] = 1
    return array
    
def get_profession_info(x):
    cases = [
        "",
        "消费品",
        "汽车.机械.制造",
        "电子.通信.硬件",
        "广告.传媒.教育.文化",
        "房地产.建筑.物业",
        "服务.外包.中介",
        "金融",
        "政府.农林牧渔",
        "交通.贸易.物流",
        "互联网.游戏.软件",
        "能源.化工.环保",    
        "制药.医疗",
    ]
    if x in cases:
        idx = cases.index(x)
    else:
        idx = 0
    array = [0] * 13
    array[idx] = 1
    return array

def get_register_channel_name(x):
    cases = [
            "",
                "洋钱罐",
        "借钱能手API合作",
        "51公积金",
        "百融API",
        "融360API",
        "拍拍贷",
        "三方机构",
        "51信用卡API",
        "应用市场：Appstore",
        "借钱能手",
        "玖富",
        "华为应用市场-借款端",
        "助力钱包",
        "百融",
        "融之家",
        "应用宝CPD"
    ]
    if x in cases:
        idx = cases.index(x)
    else:
        idx = 0
    array = [0] *17
    array[idx]=1
    return array


def get_register_source_code(x):
    cases = [
        "",
        "MOBILEWEB",
        "IOS",
        "BACK",
        "BATCH",
        "WECHAT"
    ]
    if x in cases:
        idx = cases.index(x)
    else:
        idx = 0
    array = [0] * 6
    array[idx] = 1
    return array

#=======================================
# 使用repay_history构建模型
def get_data_null_not(x):
    try:
        x = float(x)
    except:
        x = 0.0
    if x > 1e-6:
        idx = 1
    else:
        idx = 0
    array = [0]*2
    array[idx] = 1
    return array



def get_resident_province(x):
    cases = [
        ""
                "广东省",
        "江苏省",
        "浙江省",
        "上海市",
        "四川省",
        "山东省",
        "北京市",
        "福建省",
        "河南省",
        "辽宁省",
        "湖南省",
        "安徽省",
        "河北省",
        "云南省",
        "陕西省",
        "广西壮族自治区",
        "黑龙江省",
        "江西省",
        "重庆市",
        "贵州省",
        "吉林省",
        "山西省",
        "天津市",
        "海南省",
        "甘肃省",
        "青海省",
        "湖北省",
        "宁夏回族自治区",
        "新疆维吾尔自治区",
        "内蒙古自治区",
        "西藏自治区"
    ]
    if x in cases:
        idx = cases.index(x)
    else:
        idx = 0
    array = [0] *32
    array[idx]=1
    return array
def get_opening_bank(x):
    cases = [
        "建设银行",
        "工商银行",
        "中国银行",
        "招商银行",
        "邮储银行",
        "交通银行",
        "平安银行",
        "浦东发展银行",
        "中信银行",
        "兴业银行",
        "光大银行",
        "广发银行",
        "未知",
        "农业银行",
        "华夏银行",
        "民生银行",
        "邮政储蓄",
        "北京银行",
        "广东发展",
        "浦发银行"
    ]
    idx = 0
    for c in cases:
        if c in x:
            idx = cases.index(c)+1
            break
    array = [0] *21
    array[idx]=1
    return array
