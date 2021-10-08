# -*- coding: utf-8 -*-
"""
@Time     :2021/9/1 14:55
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from __future__ import absolute_import
from __future__ import division
import os, grpc, codecs
from collections import defaultdict
import numpy as np, tensorflow as tf
from scipy.special import logsumexp as lse
# from scipy.misc import logsumexp as lse
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
from tensorflow.core.framework import types_pb2
from minerva.model.environment import env
from minerva.data.feed_data import RelationEntityBatcher
from minerva.options import read_options
from utils.log import logger
from settings import config


class Predictor(object):
    def __init__(self, params):
        for key,value in params.items():
            setattr(self, key, value)
            '''
            python 中动态检查对象相关函数如下“
                hasattr(object, name) 检查 object 对象是否包含名为 name 的属性或方法。
                getattr(object, name, default=None)获取 object 对象中名为 name 的属性的属性值(属性和函数都叫做属性)。
                setattr(object, name, value)将 object 对象的 name 属性设为 value。
            类似于
            def __init__(self, a, b):
                self.a = 1
                self.b = 2
            '''
            self.save_path = None
            self.environment = env(params)
            self.entity_vocab = self.environment.entity_vocab
            self.relation_vocab = self.environment.relation_vocab
            self.type_vocab = self.environment.type_vocab
            self.entity_type_map = self.environment.entity_type_map
            self.rev_relation_vocab = self.environment.grapher.rev_relation_vocab
            self.rev_entity_vocab = self.environment.grapher.rev_entity_vocab
            self.ePAD = self.entity_vocab['PAD']
            self.rPAD = self.relation_vocab['PAD']
            self.test_rollouts = self.environment.test_rollouts
            if not os.path.exists(self.path_logger_file + "/test_beam"):
                os.mkdir(self.path_logger_file + "/test_beam")
            self.path_logger_file_ = self.path_logger_file + "/test_beam/paths"
            self.mem = (params['LSTM_layers'], 2, None, (4 if params['use_entity_embeddings'] else 2)* params['hidden_size'])

    def predict(self, online_data, print_paths=True):
        feed_dict = {}
        results = []
        answers = []
        paths = defaultdict(list)
        online_data = np.array(online_data)
        self.environment.grapher.update(online_data)  # update graph
        total_examples = online_data.shape[0]

        batcher = RelationEntityBatcher(self.batch_size,    # 将查询的相关数据封装为store [n,2]
                                        entity_vocab=self.entity_vocab,
                                        relation_vocab=self.relation_vocab,
                                        online_data=online_data)
        """
        self.environment.get_episodes(batcher)
        
        def get_episodes(self, batcher):
            for data in batcher.yield_next_batch():  # 1 episode per batch
                yield Episode(self.grapher, self.test_rollouts, data)"""
        for episode in self.environment.get_episodes(batcher):



if __name__ == "__main__":
    requests = {'data':[
        # ['CM19072750009871745', 'HasPhone'],
        # ['CM18110550000795839', 'HasCompanyName'],
        # ['CM19060250006334773', 'HasApplyAddress'],
         ['CM18102350000700439', 'HasPhone'],
         ['CM18101550000652016', 'HasCompanyName'],
         ['CM18101250000579847', 'HasApplyAddress'],
        ]}
    options = read_options()
    print(options)
    predictor = Predictor(options)
    data = requests.get('data')
    results = predictor.predict(data)
    print(results)
