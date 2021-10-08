# -*- coding: utf-8 -*-
"""
@Time     :2021/9/1 15:16
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from __future__ import absolute_import
from __future__ import division
import numpy as np
from minerva.data.grapher import RelationEntityGrapher
from utils.log import logger

class Episode(object):
    def __init__(self, graph, test_rollouts, data):
        self.grapher = graph
        self.num_rollouts = test_rollouts
        self.current_hop = 0
        # e1, r, e2, all_e2s
        start_entities, query_relation = data  # (batch_size, )
        self.no_examples = start_entities.shape[0]
        # [1,2,3]->[1,1,1,2,2,2,3,3,3]
        start_entities = np.repeat(start_entities, self.num_rollouts)  # (batch_size*num_rollouts, )
        batch_query_relation = np.repeat(query_relation, self.num_rollouts) # (batch_size*num_rollouts, )
        self.start_entities = start_entities
        self.current_entities = np.array(start_entities) # (batch_size*num_rollouts, )
        self.query_relation = batch_query_relation # (batch_size*num_rollouts, )
        # 初始化 获取当前entities的可选anctions
        next_actions = self.grapher.return_next_actions(self.current_entities)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1] # (batch*num_rollouts, max_num_actions, )
        self.state['next_entities'] = next_actions[:, :, 0] # (batch*num_rollouts, max_num_actions, )
        self.state['current_entities'] = self.current_entities # (batch_size*num_rollouts, )


class env(object):
    def __init__(self, params):
        # 初始化环境 grapher是每个实体可以选择的动作；
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             vocab_dir = params['data_input_dir'] + '/vocab')
        self.relation_vocab = self.grapher.relation_vocab
        self.entity_vocab =  self.grapher.entity_vocab
        self.type_vocab = self.grapher.type_vocab
        self.entity_type_map = self.grapher.entity_type_map
        self.relation_type_map = self.grapher.relation_type_map
        self.test_rollouts = params['test_rollouts'] #训练次数

    def get_episodes(self, batcher):
        for data in batcher.yield_next_batch():  # 1 episode per batch
            yield Episode(self.grapher, self.test_rollouts, data)