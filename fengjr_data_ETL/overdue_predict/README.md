# WOE评分模型
- 这只是本次的一个测试，原本代码参见：D:\Program Files\NWpythonwork\z-mix\永冠\overdue_predict

简单描述评分卡模型流程以及注意点：

拿到需求后，需要考虑那些东西？
1. 根据模型的目的选择目标用户群体；【本模型是根据用户的“基础信息”+“行为日志（15天前）”来预测七天后到期用户逾期概率】
   1. 待还标的选择：如果今天为20211212，days_interval=7，抽取用于预测的数据为20201219日待还的标的；
    2. 抽取方法：在"config_hql"目录下配置脚本文件，每天通过hive拉取数据，并保存一份到 "data"目录下;
2. 如何选择特征数据？
    1. 选择那些 【五类】？
       1. 基于业务理解，可选择这几类特征：用户信息特征、资产信息特征（标的特征）、风控特征、还款历史特征、app日志特征； 
          - 根据标的数据抽取：用户信息特征+标的特征+风控特征+还款历史特征 共264维；
          - 根据标的数据抽取：用户app行为日志特征【映射得到app日志的特征名 - 计算app_log特征】 共385维度；
       2. 基于用户的标的特征，将用户分为： first_period首期还款、first_overdue无历史逾期、his_overdue(有历史逾期)；
    2. 特征抽取和计算？
       - 拿到这些特征之后，需要筛选那些能用呢些不能用，需要进行"数学统计"。
       1. 遍历每个特征（即DataFrame中的每列）统计每个值出现次数，以及空值的个数。 
        2. 取Data的describe ，对于mean不为nan的值，统计每个值出现次数，选择出现次数小于5000的特征值；
        3. 通过数理统计，选择出适用于模型的特征值；
    3. 特征工程
        1. 特征筛选 （通过数理统计、经验、反复尝试筛选出适合的特征）
        2. # 样本均衡策略
        ```
        new_data_selc = pd.DataFrame()
        data_selc_pos = raw_data[raw_data['plan_status']=='CLEARED']
        data_selc_pos = data_selc_pos.sample(n=n_pos)  # 正样本采样
        data_selc_neg = raw_data[raw_data['plan_status']=='OVERDUE']
        data_selc_neg = data_selc_neg.sample(n=n_neg) # 负样本采样
        data_selc = pd.concat([data_selc_pos,data_selc_neg])
        data_selc.to_csv(DATA_PATH+os.sep+data_type+'_train_bak.csv',index=None)
    
        for fea_name in feature_select_dict['selected']:
            logger.info('select feature: %s' % fea_name)
            if fea_name not in data_selc.columns:
                 logger.error('no feture %s' % fea_name)
            else:
                new_data_selc[fea_name] = data_selc[fea_name]
        logger.info('data_selc data shape: [%d, %d]' % new_data_selc.shape)
        logger.info(new_data_selc['plan_status'].value_counts().to_string().replace("\n", "; "))
        data_selc= new_data_selc 
        ```
        3. 格式转换 缺失值填充 
        4. 标记离散特征和连续特征 【pandas怎么写】
        5. 特征分箱【需要分箱配置】
           分箱配置包括：分箱方法 分箱边界
           读取待分箱文件，对每个特征进行分箱；
            如果特征名在配置文件中：
                读取分箱处理函数，直接调用相关的分箱方法；
                    1. 自定义分箱处理函数
                    2. 连续变量 【等宽 等频 指定边界】
                    3. 卡方分箱
                    4.【离散变量】索引编码
                    5. 处理配置文件汇总未指定分享的方法,使用通用得default方法【特征idx编码-离散值】
                        记录分箱记录
        6.  计算woe table和iv值
        并且保存woe table【first_period_woe_table.json】 和 【first_period_iv_table.json】
       ```
       # 计算woe
        def feature_woe_iv(sample_data, var, target):
            eps = 0.000001
            gbi = pd.crosstab(sample_data[var], sample_data[target])+eps
            gb = sample_data[target].value_counts() + eps
            gbri = gbi/gb
            gbri['woe'] = np.log(gbri[1]/gbri[0])
            gbri['iv'] = (gbri[1]-gbri[0])*gbri['woe']
            return gbri['woe'].to_dict(), gbri['iv'].sum()
       def calc_woe_iv(sample_data, target_name):
        woe_table = {}
        iv_table = {}
        for fea_name in sample_data.columns:
            if fea_name!= target_name:
                fea_woe_dict, fea_iv = feature_woe_iv(sample_data, fea_name, 'plan_status')
                woe_table[fea_name]=fea_woe_dict
                iv_table[fea_name]=fea_iv
        return woe_table, iv_table
        ```
        7. 特征值转成woe
           使用woe_table转换woe
        ```
       # 特征woe编码    
        def feature_woe_encode(sample_data, woe_table, target_name='plan_status'):
            data_woe = pd.DataFrame()
            for fea_name, fea_data in sample_data.iteritems():
                if fea_name!=target_name:
                    fea_woe_value = fea_data.map(woe_table[fea_name])
                    data_woe[fea_name]=fea_woe_value
            data_woe[target_name]=sample_data[target_name]
            return data_woe
        ```
    4. 训练模型
    


## 本地训练版本 多了一下几步：
    特征值的数据统计分析；
    筛选特征值；
    保存woetable；
    训练模型；
## 线上版本
    直接读取特征筛选配置文件，选择模型使用的特征【这些都是训练阶段筛选的】
    读取woetable 对分箱后的特征进行编码；
    直接调用模型计算结果【训练阶段已经训练好并保存模型】