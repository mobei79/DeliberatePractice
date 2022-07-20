# coding:utf-8 -*- 
# @Time : 2019-11-14 15:43 
""" """
from neo4j import GraphDatabase
import pandas as pd
from settings import config
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 1500)

NEO4J_URI = config['nee4j']['NEO4J_URI']
NEO4J_USER = config['nee4j']['NEO4J_USER']
NEO4J_PASSWORD = config['nee4j']['NEO4J_PASSWORD']

query_template = """match (income_no:real_income_no)-[:HasIdCard]->(id:id_card)
where id.id_card in %s 
return id.id_card as id, income_no"""


class KG(object):
    """进件号、身份证、个人电话必填。依据以下子图进行抽样训练数据，主要进行实体预测（补全）
        1：从进件号的维度：两个进件号具有同一个公司名称、电话、地址中任意一个，其他的应该相同
        2：以身份证号维度：同一个身份证下多个进件号具有相同个人信息（身份证、户籍地址、居住地址、设备地址、ip、紧急电话、个人电话）
    """

    def __init__(self):
        self._driver = GraphDatabase().driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    @staticmethod
    def query_fn(tx, q):
        return tx.run(q)

    def query(self, q):
        with self._driver.session() as sess:
            rst = sess.read_transaction(self.query_fn, q)
            return rst


kg = KG()


def query(id_cards, props):
    """
    批量查询所有id的所有属性
    :param id_cards:
    :type id_cards: []
    :param props:
    :type props:
    :return: df:['id', 'income_no', 'income_tm'] + props
    :rtype:
    """
    query = query_template % id_cards
    rst = kg.query(query)
    data = rst.data()

    info = []
    for re in data:
        data = re

        info.append([re['id'], re['income_no']['real_income_no'], re['income_no']['income_no_create_tm']] \
                    + [re['income_no'][p] if re['income_no'][p] != 'None' else None for p in props])
    df = pd.DataFrame(info, columns=['id', 'income_no', 'income_tm'] + props)
    df.loc[:, 'income_tm'] = pd.to_datetime(df['income_tm'])
    # print(df)
    return df


if __name__ == '__main__':
    ids = ["++tBtikJS9WFLyixAAul8vXzzflj1eDAI34huHLnB88=",'+0Y3oMLnSwE9r4MepU0+ZUszgGM6OjZYzVrah/daHp4=']
    # props = ['profession_station', 'start_work_tm']
    props = ['credit_card_number',
            'company_phone',
            'famliy_relationship',
            'resident_address',
            'highest_eduction',
            'company_address',
            'workmate_phone',
            'workmate_name',
            'family_name',
            'work_relationship',
            'family_phone',
            'profession_station',
            'company_name',
            'business_source',
            'is_married',
            'company_town',
            'start_work_tm',
            'channel_code',
            'loan_use',
            'job_salary',
            'company_city',
            'resident_town',
            'income_address',
            'device_address',
            'company_province',
            'mobile_prov',
            'mobile_encrypt',
            'mobile_city',
            'company_name',
            'resident_address',
            'company_address',
            'company_name',
            'resident_address',
            'company_address']
    query(ids, props)

if __name__ == '__main__':
    print(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)