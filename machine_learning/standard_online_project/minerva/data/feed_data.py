# -*- coding: utf-8 -*-
"""
@Time     :2021/9/1 22:44
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import numpy as np

from utils.log import logger

class RelationEntityBatcher():
    def __init__(self, batch_size, entity_vocab, relation_vocab, online_data=None):
        self.batch_size = batch_size
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.create_triple_store(online_data)

    def create_triple_store(self, input):
        self.store = np.array(input)
        self.store[:, 0] = np.vectorize(self.entity_vocab.get)(self.store[:, 0])
        self.store[:, 1] = np.vectorize(self.relation_vocab.get)(self.store[:, 1])
        self.store = self.store.astype('int32')
        # print(self.store)

    # 每次预测传入新的进件，我们查询新进件关联的节点，封装到store中，行为此次添加的节点，第一列为节点，第二列为边或称之为关系
    def yield_next_batch(self):
        print("***&&&&&&&&&&&&&&&&&&&&&&&&")
        self.store = np.array([[26,13] ,[38,10] ,[51,7]])
        print(self.store)
        remaining_triples = self.store.shape[0]     # shape[0] 为此次新添加的节点数目
        current_idx = 0
        # 保证所有测试样本 全部被测试到
        while True:
            if remaining_triples == 0:
                return
            if remaining_triples - self.batch_size > 0: # 如果传入的数目大于265
                batch_idx = np.arange(current_idx, current_idx + self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                batch_idx = np.arange(current_idx, self.store.shape[0])
                remaining_triples = 0
                print(batch_idx)
            batch = self.store[batch_idx, :]
            print(batch)
        #     e1 = batch[:, 0]
        #     r = batch[:, 1]
        #     assert e1.shape[0] == r.shape[0]
        #     yield e1, r
    def testfuy(self):
        print("***************")
        self.store = np.array([[26, 13], [38, 10], [51, 7]])
        print(self.store)
        remaining_triples = self.store.shape[0]  # shape[0] 为此次新添加的节点数目
        current_idx = 0
        print("***************")
if __name__ == "__main__":
    import json
    requests = {'data': [
        ['CM19072750009871745', 'HasPhone'],
        ['CM18110550000795839', 'HasCompanyName'],
        ['CM19060250006334773', 'HasApplyAddress'],
    ]}
    online_data = requests["data"]


    ad = np.array(online_data)
    add = np.arange(0, 2)
    ac = ad[add,:]
    print(ad)
    print(add)
    print(ac)







    # batch_size = 256
    # entity_vocab = json.load(open('D:\Program Files\pythonwork\DeliberatePractice\machine_learning\standard_online_project\datasets\data_preprocessed\\raw_data_72138\\vocab/entity_vocab.json'))
    # relation_vocab = json.load(open('D:\Program Files\pythonwork\DeliberatePractice\machine_learning\standard_online_project\datasets\data_preprocessed\\raw_data_72138\\vocab/relation_vocab.json'))
    # re1 = RelationEntityBatcher(batch_size, entity_vocab, relation_vocab, online_data)
    # print(re1.store.shape[0])
    #
    # re1.yield_next_batch()
    # print("end")

    # batch_i = np.arange(1, 256)
    # print(batch_i)


    #
    # data = requests["data"]
    # stroe = np.array(data)
    # print(stroe)
    # print(stroe.shape)
    #
    #
    #
    # import json
    # relation_vocab = json.load(open('D:\Program Files\pythonwork\DeliberatePractice\machine_learning\standard_online_project\datasets\data_preprocessed\\raw_data_72138\\vocab/relation_vocab.json'))
    # print(type(relation_vocab))
    # print(relation_vocab)
    # stroe[:, 1] = np.vectorize(relation_vocab.get)(stroe[:, 1])
    # print(stroe[:,1])