# coding:utf-8 -*- 
# @Time : 2019-11-14 15:43 
""" """
from neo4j import GraphDatabase

NEO4J_URI = "bolt://10.10.203.132:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "kgfengjr"

props = {'HasCompanyPhone': 'phone',
         'HasCompanyAddress': 'address',
         'HasCompanyName': 'company_name',
         'HasDeviceAddress': 'device_address',
         'HasApplyAddress': 'address',
         'HasIP': 'device_ip',
         'HasEmergencyPhone': 'phone',
         'HasHomeAddress': 'address',
         'HasPhone': 'phone',
         'HasIdCard': 'id_card'}

# query_template = "match (no1:real_income_no{real_income_no:'%s'})-[r]->(no2) where type(r) <>'HasContact' return no1, type(r) as r, no2"
# income_no 的整个相关的子图
query_template = """match 
(n1:real_income_no{real_income_no:'%s'})-[:HasIdCard]->(:id_card)<-[:HasIdCard]-(n2:real_income_no)
with collect(n1)+collect(n2) as nodez
UNWIND nodez as no1
match (no1)-[r]->(no2) where type(r) <>'HasContact'
return no1.real_income_no as no1, type(r) as r, no2
"""


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


def query_data(income_no):
    query = query_template % income_no
    # print('query=\n{}'.format(query))
    rst = kg.query(query)
    data = rst.data()
    actions = []

    for record in data:
        e1 = record['no1']
        r = record['r']
        e2 = record['no2'][props[r]]
        actions.append([e1, r, e2])
    return actions


if __name__ == '__main__':
    print(query_data('CM19111450015633330'))
