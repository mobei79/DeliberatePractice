# -*- coding: utf-8 -*-
# !/usr/bin/env python
import pandas as pd
#from sklearn.impute import SimpleImputer
import numpy as np
from config.feature_config import *
import os, logging, pprint
import json,chardet
import datetime

# raw feature total : 656

DATA_PATH = 'data'
OUT_DATA_PATH = "data_for_model"
LOG_PATH = 'log'
CONF_PATH = 'config'
cur_date = datetime.date.today()
cur_date_str = cur_date.strftime("%Y-%m-%d")

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.FileHandler(LOG_PATH+os.sep+"feature_process.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# 列名标准化
def rename_col_name(df):
    col_names = df.columns.tolist()
    rename_map = {}
    for col_name in col_names:
        new_name = col_name.split('.')[-1]
        rename_map[col_name]=new_name
    df.rename(columns=rename_map, inplace=True)

# 日期数据转成与某一天的时间间隔
def date_to_days(feature_data, target_date):
    result = pd.to_datetime(feature_data)-pd.to_datetime(target_date)
    result = result.dt.days
    return result

# 删除指定列
def del_feature(sample_data, feature_names):
    for feature_name in feature_names:
        if feature_name in sample_data.columns:
            sample_data.drop(feature_name, axis=1, inplace=True)
    
# 删除特征缺失占比p以上的样本
def del_sample_by_miss_prop(sample_data, miss_prop):
    result = pd.DataFrame()
    fearture_count = sample_data.shape[1]
    sample_data['nan_prop'] = sample_data.apply(lambda x: x.isna().sum()/fearture_count, axis=1)
    sample_data.drop(sample_data[sample_data.nan_prop > miss_prop].index, inplace=True)
    del sample_data['nan_prop']
    sample_data.reset_index(drop=True, inplace=True)
    return sample_data

# 删除特征值在指定范围的样本
def del_sample_custom(sample_data, feature_name, value_range):
    if len(value_range)==2:
        result_data = sample_data[(sample_data[feature_name]<value_range[0]) | (sample_data[feature_name]>feature_name[1])]
    if len(value_range)==1:
        result_data = sample_data[sample_data[feature_name]!=value_range[0]]
    return result_data

# 自定义分箱处理逻辑
def custom_binning_process(feature_name, feature_data, func):
    binning_map = {}
    feature_data.fillna("NaN",inplace=True)
    for value in feature_data.unique():
        binning_map[value] = func(value).index(1)
    result = feature_data.apply(lambda x: binning_map[x])
    if result.isna().sum()>0:
        logger.warning('feature [%s] binning result NULL count: %d' %  (feature_name, result.isna().sum()))
    return result
    
# 特征分箱-连续值
def binning_continuous(feature_data, bins=8,labels=None, method='equal_width'):

    if method=='custom_boundary':
        if len(bins)>=1:
            result, out_bins = pd.cut(feature_data,bins=bins,labels=labels,retbins=True)
        else:
            logger.error('feature have no binning boundary!')
            return None, None
    if method=='equal_width':
        result, out_bins = pd.cut(feature_data,bins=bins,labels=labels, retbins=True)
    if method=='equal_size':
        result, out_bins=pd.qcut(feature_data,q=bins,labels=labels,retbins=True)
    return result, out_bins.tolist()

# 计算卡方值
def calc_chi(con_table):
    assert(con_table.ndim==2)
    row_count = con_table.sum(axis=1)
    col_count = con_table.sum(axis=0)
    total_count = con_table.sum()
    e_table = np.ones(con_table.shape)*col_count/total_count
    e_table = (e_table.T*row_count).T
    sqr_table = (con_table-e_table)**2/e_table
    sqr_table[e_table==0] = 0
    chi_value = sqr_table.sum()
    return chi_value

# 卡方分箱
def chi_merge(sample_data, fea_name, target_name, max_groups=10, threshold=None):
    freq_table_df = pd.crosstab(sample_data[fea_name], sample_data[target_name])
    freq_table = freq_table_df.values
    cut_offs = freq_table_df.index.values

    if max_groups is None:
        if threshold is None:
            # TODO
            pass
    while True:
        min_chi_v = None
        min_bin_idx = None
        # 遍历找到最小卡方值的两组相邻分箱
        for i in range(len(freq_table)-1):
            chi_v = calc_chi(freq_table[i:i+2])
            if min_chi_v is None or chi_v<min_chi_v:
                min_chi_v = chi_v
                min_bin_idx = i

        # 如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
        if max_groups is not None and max_groups<len(freq_table) or (threshold is not None and min_chi_v < threshold):
            tmp = freq_table[min_bin_idx]+freq_table[min_bin_idx+1]
            freq_table[min_bin_idx] = tmp
            freq_table = np.delete(freq_table, min_bin_idx+1, 0)
            cut_offs = np.delete(cut_offs, min_bin_idx+1, 0)
        # 否则停止合并
        else:
            break
    cut_offs = sorted(cut_offs)
    num_groups = len(cut_offs)
    binning_map = {}
    for x in sample_data[fea_name].unique():
        if pd.isna(x):
            # binning_map[x] = 'group{}'.format(num_groups+1)
            binning_map[x] = num_groups+1
        if x<cut_offs[0]:
            binning_map[x] = 1
            # binning_map[x] = 'group{}'.format(1)
        if x>=cut_offs[-1]:
            binning_map[x] = 1
            # binning_map[x] = 'group{}'.format(1)
        for i in range(1, num_groups):
            if cut_offs[i-1]<= x <cut_offs[i]:
                binning_map[x] = i
                # binning_map[x] = 'group{}'.format(i)
    result = sample_data[fea_name].map(binning_map)

    return result, cut_offs

# 特征idx编码-离散值
def encode_discrete(feature_data,encode_map=None):
    #feature_data = feature_data.map({'nan':'NaN'})
    fea_values = feature_data.unique().tolist()
    feature_data = feature_data.replace({'nan':'NaN'})
    fea_values = feature_data.unique().tolist()
    new_map = {}
    for k,v in encode_map.items():
        try:
            new_map[k.encode('utf-8')]=v
        except:
            print('编码错误：')
            print(k)
            print(k.encode('utf-8'))
    for v in fea_values:
        if v not in new_map:
            print("encode 缺失##########"+v)
            new_map[v]=1
    result = feature_data.replace(new_map)
    return result, new_map

# 计算woe
def feature_woe_iv(sample_data, var, target):
    eps = 0.000001
    gbi = pd.crosstab(sample_data[var], sample_data[target])+eps
    gb = sample_data[target].value_counts() + eps
    gbri = gbi/gb
    gbri['woe'] = np.log(gbri[1]/gbri[0])
    gbri['iv'] = (gbri[1]-gbri[0])*gbri['woe']
    return gbri['woe'].to_dict(), gbri['iv'].sum()

# 计算woe table和iv值
def calc_woe_iv(sample_data, target_name):
    woe_table = {}
    iv_table = {}
    for fea_name in sample_data.columns:
        if fea_name!= target_name:
            fea_woe_dict, fea_iv = feature_woe_iv(sample_data, fea_name, 'plan_status')
            woe_table[fea_name]=fea_woe_dict
            iv_table[fea_name]=fea_iv
    return woe_table, iv_table

# 特征woe编码    
def feature_woe_encode(sample_data, woe_table):
    data_woe = pd.DataFrame()
    for fea_name, fea_data in sample_data.iteritems():
        fea_values = fea_data.unique().tolist()
        #print(fea_name)
        #print(fea_values)
        for v in fea_values:
            if str(v) not in woe_table[fea_name]:
                print('woe缺失：'+str(v))
                woe_table[fea_name][str(v)]=0
        cur_fea_woe_table = {}
        for k, v in woe_table[fea_name].items():
            cur_fea_woe_table[int(float(k))] = v
        fea_woe_value = fea_data.map(cur_fea_woe_table)
        data_woe[fea_name]=fea_woe_value
    return data_woe

def feature_binning(feature_data, binning_map,boundary_record):
    feature_binning_data = pd.DataFrame()
    for fea_name, item_data in feature_data.iteritems():
        print(fea_name)
        # 配置文件中指定分箱方法
        if fea_name in binning_map:
            binning_method = binning_map[fea_name][0]
            #  1.自定义分箱处理函数，直接调用相关方法分箱
            if binning_method=='custom_func':  
                func = binning_map[fea_name][1][2]
                binning_result = custom_binning_process(fea_name, item_data, func)

            #  2.【连续变量】等宽，等频，指定边界
            elif binning_method in ['equal_width','equal_size','custom_boundary']: 
                item_data = item_data.fillna(item_data.min())
                try: 
                    bins = boundary_record[fea_name][1]   # 读入已保存好的分箱边界
                    bins[0] = float('-inf')
                    bins[-1] = float('inf')
                except:
                    logger.error('feature [%s] have no binning boundary!' % fea_name)
                if len(binning_map[fea_name][1])==1:
                    labels = [i for i in range(len(bins)-1)]
                else:
                    labels = binning_map[fea_name][1][2]
                binning_result, _ = binning_continuous(item_data, bins=bins, labels=labels, method='custom_boundary')
            #  3.卡方分箱
            elif binning_method=='chi_merge':
                #  读入分箱边界
                try: 
                    bins = boundary_record[fea_name][1]   # 读入已保存好的分箱边界
                    #bins[0] = float('-inf')
                    #bins[-1] = float('inf')
                except:
                    logger.error('feature [%s] have no binning boundary!' % fea_name)
                labels = [i for i in range(len(bins)-1)]
                item_data = item_data.fillna(-99999)
                binning_result, _ = binning_continuous(item_data, bins=bins, labels=labels, method='custom_boundary')
            #  4.【离散变量】索引编码
            elif binning_method == 'index_encode':
                #  读入encode map
                try: 
                    encode_map = boundary_record[fea_name][1]   # 读入已保存好的分箱边界
                    if encode_map==None:
                        logger.error('feature [%s] have no encode map!' % fea_name)
                    
                except:
                    logger.error('feature [%s] have no encode map!' % fea_name)
                item_data = item_data.apply(lambda x: str(x))
                binning_result, _ = encode_discrete(item_data,encode_map)

        # 配置文件中未指定分箱方法
        else:
            binning_method = 'default'
            if item_data.dtype=='object':
                # 记录encode_map
                binning_result, _ = encode_discrete(item_data)
            else:
                # 记录分箱边界
                binning_result, _ = binning_continuous(item_data)
        if binning_result.isna().sum()>0:
            print('null#####################'+fea_name, binning_result.isna().sum())
        feature_binning_data[fea_name]=binning_result
        logger.info('feature binning: [%s], method: [%s]' % (fea_name, binning_method))
    return feature_binning_data

def feature_process_pre(raw_feature_data_file):

    logger.info('start process feature data: %s' % raw_feature_data_file)
    date_suffix = '_'.join(raw_feature_data_file.split('.')[0].split('_')[-2:])
    #date_suffix = 'testdata'
    
    # step1: 读入数据
    raw_data = pd.read_csv(raw_feature_data_file)
    logger.info('read raw data shape: [%d, %d]' % raw_data.shape)
    

    # step2: 样本按照类别拆分：首期待还、无历史逾期待还、其他待还
    raw_data.drop(raw_data[raw_data.plan_status.isin(["PRE_CLEARED"])].index, inplace=True) # 去除提前还款的标的
    first_period_data = raw_data[raw_data['periods_no']==1]
    first_overdue_data = raw_data[(raw_data['periods_no']!=1) & (raw_data['his_overdue_cnt']==0)]
    his_overdue_data = raw_data[(raw_data['periods_no']!=1) & (raw_data['his_overdue_cnt']!=0)]
    logger.info('first_period_data data shape: [%d, %d]' % first_period_data.shape)
    logger.info('first_overdue_data data shape: [%d, %d]' % first_overdue_data.shape)
    logger.info('his_overdue_data data shape: [%d, %d]' % his_overdue_data.shape)

    # step3: 样本筛选
    # merge_data_selc = del_sample_custom(merge_data_selc, 'periods_no', [0]) # 去除期数为0的样本
    
    # step3: 特征筛选
    first_period_data_selc = pd.DataFrame()
    first_overdue_data_selc = pd.DataFrame()
    his_overdue_data_selc = pd.DataFrame()
    first_period_fea_select_dict = first_overdue_feature_select_v1
    first_overdue_fea_select_dict = first_overdue_feature_select_v1
    his_overdue_fea_select_dict = his_overdue_feature_select_v1

    for fea_name in first_period_fea_select_dict['selected']:
        first_period_data_selc[fea_name] = first_period_data[fea_name]
    for fea_name in first_overdue_fea_select_dict['selected']:
        first_overdue_data_selc[fea_name] = first_overdue_data[fea_name]
    for fea_name in his_overdue_fea_select_dict['selected']:
        his_overdue_data_selc[fea_name] = his_overdue_data[fea_name]

    first_period_data_selc.drop(['plan_status'], axis=1, inplace=True)
    first_overdue_data_selc.drop(['plan_status'], axis=1, inplace=True)
    his_overdue_data_selc.drop(['plan_status'], axis=1, inplace=True)
    first_period_data_selc.reset_index(drop=True, inplace=True)
    first_overdue_data_selc.reset_index(drop=True, inplace=True)
    his_overdue_data_selc.reset_index(drop=True, inplace=True)
    logger.info('first_period_data_selc shape: [%d, %d]' % first_period_data_selc.shape)
    logger.info('first_overdue_data_selc shape: [%d, %d]' % first_overdue_data_selc.shape)
    logger.info('his_overdue_data_selc shape: [%d, %d]' % his_overdue_data_selc.shape)

    # step4: 保存特征筛选后的数据
    out_file_1 = DATA_PATH+os.sep+'first_period_fea_%s.csv' % date_suffix
    out_file_2 = DATA_PATH+os.sep+'first_overdue_fea_%s.csv' % date_suffix
    out_file_3 = DATA_PATH+os.sep+'his_overdue_fea_%s.csv' % date_suffix
    first_period_data_selc.to_csv(out_file_1, index=None)
    first_overdue_data_selc.to_csv(out_file_2, index=None)
    his_overdue_data_selc.to_csv(out_file_3, index=None)
    logger.info('output data file: %s' % out_file_1)
    logger.info('output data file: %s' % out_file_2)
    logger.info('output data file: %s' % out_file_3)
    first_period_data_selc = first_period_data_selc.drop(['user_id', 'loan_id'], axis=1, inplace=False)
    first_overdue_data_selc = first_overdue_data_selc.drop(['user_id', 'loan_id'], axis=1, inplace=False)
    his_overdue_data_selc = his_overdue_data_selc.drop(['user_id', 'loan_id'], axis=1, inplace=False)
    
    '''
    # step4: 日期格式特征转换
    for fea_name, target_date in date_trans_dict.items():
        feature_days = date_to_days(merge_data_selc[fea_name], target_date)
        merge_data_selc['fea_name'] = feature_days
        logger.info('date value transform to days: [%s], target data: [%s]' % (fea_name, target_date))


    # step5: 缺失值填充
    for fea_name,item_data in merge_data_selc.iteritems():
        if fea_name in feature_filling_dict:
            method = feature_filling_dict[fea_name]
            # TODO
        else:
            impute_value = merge_data_selc[fea_name].mode()[0]   # 填充值：众数
            merge_data_selc[fea_name] = merge_data_selc[fea_name].fillna(impute_value)
        logger.info('filling missing value: %s' % fea_name)
    '''

    # step6: 特征分箱
    # 选择分箱配置
    first_period_bins_config = feature_bins_v1
    first_overdue_bins_config = feature_bins_v1
    his_overdue_bins_config = feature_bins_v1
    # 读取已保存的分箱边界
    f1 = open(CONF_PATH+os.sep+'first_period_binning_boundary.json', 'r')
    f2 = open(CONF_PATH+os.sep+'first_overdue_binning_boundary.json', 'r')
    f3 = open(CONF_PATH+os.sep+'his_overdue_binning_boundary.json', 'r')
    first_period_bins_boundary = json.load(f1)
    first_overdue_bins_boundary = json.load(f2)
    his_overdue_bins_boundary= json.load(f3)
    f1.close()
    f2.close()
    f3.close()
    first_period_binning_data = feature_binning(first_period_data_selc,first_period_bins_config,first_period_bins_boundary)
    logger.info("first_period_binning_data shape: [%d, %d]" % first_period_binning_data.shape)
    first_overdue_binning_data = feature_binning(first_overdue_data_selc,first_overdue_bins_config,first_overdue_bins_boundary)
    logger.info("first_overdue_binning_data shape: [%d, %d]" % first_overdue_binning_data.shape)
    his_overdue_binning_data = feature_binning(his_overdue_data_selc,his_overdue_bins_config,his_overdue_bins_boundary)
    logger.info("first_overdue_binning_data shape: [%d, %d]" % first_overdue_binning_data.shape)
    f1 = OUT_DATA_PATH+os.sep+'first_period_binning_data_%s.csv' % date_suffix
    first_period_binning_data.to_csv(f1,  index=None)
    logger.info("export first_period_binning_data file to: %s" % f1)
    f2 = OUT_DATA_PATH+os.sep+'first_overdue_binning_data_%s.csv' % date_suffix
    first_overdue_binning_data.to_csv(f2,  index=None)
    logger.info("export first_overdue_binning_data file to: %s" % f2)
    f3 = OUT_DATA_PATH+os.sep+'his_overdue_binning_data_%s.csv' % date_suffix
    his_overdue_binning_data.to_csv(f3,  index=None)
    logger.info("export his_overdue_binning_data file to: %s" % f3)
    #print(first_period_binning_data.head())
    # exit()
    first_period_binning_data.to_csv('firp_binning.csv')
    # step7: 读取 woe table
    f1 = open(CONF_PATH+os.sep+'first_period_woe_table.json', 'r')
    f2 = open(CONF_PATH+os.sep+'first_overdue_woe_table.json', 'r')
    f3 = open(CONF_PATH+os.sep+'his_overdue_woe_table.json', 'r')
    woe_table_1 = json.load(f1)
    woe_table_2 = json.load(f2)
    woe_table_3 = json.load(f3)
    f1.close()
    f2.close()
    f3.close()

    # step8: 特征woe编码
    first_period_woe_data = feature_woe_encode(first_period_binning_data, woe_table_1)
    logger.info("first_period_binning_data transform to woe.")
    first_overdue_woe_data = feature_woe_encode(first_overdue_binning_data, woe_table_2)
    logger.info("first_overdue_binning_data transform to woe.")
    his_overdue_woe_data = feature_woe_encode(his_overdue_binning_data, woe_table_3)
    logger.info("his_overdue__binning_data transform to woe.")

    # step9: 保存woe格式数据
    f1 = OUT_DATA_PATH+os.sep+'first_period_woe_data_%s.csv' % date_suffix
    first_period_woe_data.to_csv(f1,  index=None)
    logger.info("export first_period_woe_data file to: %s" % f1)

    f2 = OUT_DATA_PATH+os.sep+'first_overdue_woe_data_%s.csv' % date_suffix
    first_overdue_woe_data.to_csv(f2, index=None)
    logger.info("export first_overdue_woe_data file to: %s" % f2)

    f3 = OUT_DATA_PATH+os.sep+'his_overdue_woe_data_%s.csv' % date_suffix
    his_overdue_woe_data.to_csv(f3, index=None)
    logger.info("export first_overdue_woe_data file to: %s" % f3)

    logger.info('end process feature data.')

if __name__ == "__main__":

    file_name = ''
    with open('date_align.record', 'r') as f:
        file_name = f.readline().strip()
    if file_name == '':
        logger.error("raw feature file not found!")
    raw_data_file = DATA_PATH+os.sep+file_name
    #raw_data_file = DATA_PATH+os.sep+'fo_train.csv'
    feature_process_pre(raw_data_file)
