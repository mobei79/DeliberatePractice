# -*- coding: utf-8 -*-
"""
@Time     :2021/9/1 15:24
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from collections import defaultdict
import numpy as np
import csv,json
from utils.log import logger
from minerva.data.database import query_data

class RelationEntityGrapher:
    def __init__(self, triple_store, max_num_actions, vocab_dir):
        """
        构建agent的游走图（矩阵），即每个实体可以选择的关系（action）和到达的新实体（状态）
        Args:
            triple_store (): 整个graph三元组的文件路径    我们选取了48 6430组三元组；双向
            relation_vocab ():  关系词汇 {关系：编号}
            entity_vocab ():    实体词汇  {实体：编号}
            max_num_actions (): 200  最大的训练次数
        """
        # self.entity_vocab_dir = vocab_dir + '/entity_vocab.json'
        self.entity_vocab = json.load(open(vocab_dir + '/entity_vocab.json'))   # {实体：实体编号}
        self.relation_vocab = json.load(open(vocab_dir + '/relation_vocab.json'))   #{关系类型：关系类型编号}  HasCompanyAddress:5
        self.type_vocab = json.load(open(vocab_dir + '/type_vocab.json'))   # {实体类型：实体类型编号}
        # 实体的编码 和 实体类型编码 之间的映射关系
        self.entity_type_map = {int(k): v for k, v in json.load(open(vocab_dir + '/entity_type_map.json')).items()}  # {实体编号：实体类型编号}
        self.relation_type_map = json.load(open(vocab_dir + '/relation_type_map.json'))     # {关系类型：实体类型} 关联关系
        assert len(self.entity_type_map) == len(self.entity_vocab)
        # logger.info(
        #     'Vocab guo files loaded... entity-vocab length={},relation-vocab length={},type-vocab length={},entity-type-map length={},relation-type-map length={}' \
        #     .format(len(self.entity_vocab),
        #             len(self.relation_vocab),
        #             len(self.type_vocab),
        #             len(self.entity_type_map),
        #             len(self.relation_type_map)))
        print(
            'Vocab guo files loaded... entity-vocab length={},relation-vocab length={},type-vocab length={},entity-type-map length={},relation-type-map length={}' \
            .format(len(self.entity_vocab),
                    len(self.relation_vocab),
                    len(self.type_vocab),
                    len(self.entity_type_map),
                    len(self.relation_type_map)))
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        self.triple_store = triple_store
        # 每个实体 的出度 r 和 e
        self.store = defaultdict(list)
        # max_num_actions=200  (Vocab_size, max_num_actions, 2)
        self.array_store = np.ones((len(self.entity_vocab),max_num_actions,2),dtype=np.dtype("int32"))
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD
        self.masked_array_store = None
        self.rev_relation_vocab = dict([(v, k) for k, v in self.relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in self.entity_vocab.items()])
        self.create_graph()
        logger.info('Graph loaded... shape={}'.format(self.array_store.shape))

    def create_graph(self):
        """
        每一个实体 构建 200个三元组  (len(entity_vocab), max_num_actions, 2(r, e2))
        即：每个实体构建最多200个候选action
        Returns:
        """
        # 先生成 {e1:[(r,e2),(r,e2),...],...}的dict
        with open(self.triple_store) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                e1 = self.entity_vocab[line[0]]
                r = self.relation_vocab[line[1]]
                e2 = self.entity_vocab[line[2]]
                self.store[e1].append((r, e2)) # {'e1':[(r, e2),(r, e2),...]}
        # 再构建 state-action-state 矩阵 (Vocab_size, max_num_actions, 2)
        # 三维矩阵 1维是实体个数；2维是实体可以去到的节点也就是actions；3维是action到的节点和action途经的关系
        for e1 in self.store:
            num_actions = 1
            # (e1, NO_OP, e1)
            self.array_store[e1, 0, 1] = self.relation_vocab["NO_OP"]  # relation_vocab["NO_OP"] =1
            self.array_store[e1, 0, 0] = e1
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]:
                    break#200个的时候结束
                self.array_store[e1, num_actions, 0] = e2
                self.array_store[e1, num_actions, 1] = r
                num_actions +=1
        del self.store
        self.store = None

    def add_capacity(self, e, type):
        """add entity into vocab, r into entity_map vocab, add capacity by 1.1 """
        self.entity_vocab[e] = e_id = len(self.entity_vocab)
        self.entity_type_map[e_id] = self.type_vocab[type]
        logger.debug('entity vocab updated length={} entity type map length={}: entity={} id={} type={}' \
                     .format(len(self.entity_vocab), len(self.entity_type_map), e, e_id, self.type_vocab[type]))
        assert len(self.entity_type_map) == len(self.entity_vocab)
        # add capacity
        if self.array_store.shape[0] <= e_id + 1:
            capacity = int(self.array_store.shape[0] * 0.1)
            capa = np.ones((capacity, self.array_store.shape[1], 2), dtype=np.dtype('int32'))
            capa[:, :, 0] *= self.ePAD
            capa[:, :, 1] *= self.rPAD
            self.array_store = np.vstack((self.array_store, capa))
            logger.info('array store capacity added from {} to {}'.format(e_id + 1, self.array_store.shape[0]))
        # (e1, NO_OP, e1)
        self.array_store[e_id, 0, 0] = e_id
        self.array_store[e_id, 0, 1] = self.relation_vocab['NO_OP']
        return e_id

    def update(self, data):
        """add non-existed income number into graph
        :param data: online query data e1, r
        :type data: ndarray (None, 2)
        :return:
        :rtype:
        """
        has_new = False
        for e1, _ in data:
            if e1 not in self.entity_vocab:
                has_new = True
                # query_data 需要查询 进件号同idCard关联的所有进件，以及这些进件的关系。
                new_no_actions = query_data(e1) # new_no_actions data type= [[node1,r ,node2],[]]
                if new_no_actions:
                    self.add_capacity(e1, "real_income_no")
                    num_actions = 1
                    for no1, r, no2 in new_no_actions:
                        # no1_id, no2_id = 0, 0
                        if no1 not in self.entity_vocab:
                            nod1_id = self.add_capacity(no1, "real_income_no")
                        else:
                            nod1_id = self.entity_vocab[no1]
                        if no2 not in self.entity_vocab:
                            nod2_id = self.add_capacity(no2, self.relation_type_map[r])
                        else:
                            nod2_id = self.entity_vocab[no2]

                        action_id = (self.array_store[nod1_id,:,0] == self.ePAD).argmax(axis=-1) # 获取nod1_id 可选动作的 相邻节点的最大值的索引。
                        if action_id < self.array_store.shape[1]:
                            self.array_store[nod1_id, action_id, 0] = self.entity_vocab[no2]
                            self.array_store[nod1_id, action_id, 1] = self.relation_vocab[r]

                        action_id = (self.array_store[nod2_id, :, 0] == self.ePAD).argmax(axis=-1)  # 获取nod1_id 可选动作的 相邻节点的最大值的索引。
                        if action_id < self.array_store.shape[1]:
                            self.array_store[nod2_id, action_id, 0] = self.entity_vocab[no1]
                            self.array_store[nod2_id, action_id, 1] = self.relation_vocab[r]
        if has_new:
            l = len(self.rev_relation_vocab)
            new = {v:k for k,v in self.entity_vocab.items}
            self.rev_relation_vocab.update(new)
            logger.info('rev_entity_vocab updated from {} to {}'.format(l, len(self.rev_entity_vocab)))







if __name__ == "__main__":
    # grapher = RelationEntityGrapher(triple_store="D:\Program Files\pythonwork\DeliberatePractice\machine_learning\standard_online_project\datasets\data_preprocessed\\raw_data_72138\graph.txt",
    #                     max_num_actions=200,
    #                     vocab_dir= 'D:\Program Files\pythonwork\DeliberatePractice\machine_learning\standard_online_project\datasets\data_preprocessed\\raw_data_72138/vocab'
    #                     )
    a = np.ones((4,3,2),dtype=np.dtype('int32'))
    a[0, 0, 0] = 111
    a[0, 0, 1] = 112
    a[0, 1, 0] = 121
    a[0, 1, 1] = 122
    a[0, 2, 0] = 131
    a[0, 2, 1] = 132
    a[1, 0, 0] = 211
    a[1, 0, 1] = 212
    a[1, 1, 0] = 221
    a[1, 1, 1] = 222
    a[1, 2, 0] = 231
    a[1, 2, 1] = 232
    # a[1, 0, 0] = 3
    # a[2, 0, 0] = 3
    # a[3, 0, 0] = 4
    # a[0, 0, 0] = "e1"
    # a[1, 0, 0] = "e2"
    # a[2, 0, 0] = "e3"
    # a[3, 0, 0] = "e4"
    print(a.shape[0])
    # print(a)
    # print(a[1,:,:])
    # print("**************")
    # print(a[1,1])