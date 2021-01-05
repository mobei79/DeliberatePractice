# -*- coding: utf-8 -*-
"""
@Time     :2020/11/11 15:46
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community


class Model(object):
    """docstring for Model"""

    def __init__(self, fieldFileName, trainFileName):
        self.fieldFileName = fieldFileName
        self.trainFileName = trainFileName

    def readField(self, fieldFileName):
        try:
            field = np.loadtxt('./' + fieldFileName + '.txt', dtype=str)
            print('shape of field=%f' % field.shape[0])
            return field
        except Exception as e:
            raise e

    def readData(self, trainFileName, delimiter, field):
        try:
            data = pd.read_csv('./' + trainFileName + '.csv', delimiter=delimiter, dtype=None, names=field)
            print(data.shape)
            print(data.columns)
            print(data.iloc[0, :])
            return data
        except Exception as e:
            raise e


if __name__ == '__main__':
    # fieldFileName='field'
    # print 'fieldFileName=%s'%fieldFileName
    # trainFileName='data'
    # print 'trainFileName=%s'%trainFileName

    # m = Model(fieldFileName,trainFileName)

    # field = m.readField(fieldFileName)

    # print field

    # delimiter='\t'
    # data = m.readData(trainFileName,delimiter,field)
    # data.drop(['dt','type'],axis=1,inplace=True)
    # data['w']=1
    # graph = igraph.load('./data.csv',format='ncol',directed=False,names=field)
    # graph = igraph.Graph()
    # for index,row in data.iterrows():
    #	#print row
    #	if row.type!=2:
    #		graph.add_vertex(row.from_id)
    #		graph.add_vertex(row.to_id)
    #		graph.add_edge(row.from_id,row.to_id)
    #	break
    # graph = igraph.load('./data.csv',format='edgelist')
    # edgelist = []
    # weights = []
    # for i in range(data.shape[0]):
    #	edge = (data.loc[i,'from_id'], data.loc[i,'to_id'])
    #	if edge not in edgelist:
    #		edgelist.append(edge)
    #		weights.append(1)
    #	else:
    #		weights[edgelist.index(edge)] += 1
    # graph.add_edges(edgelist)
    # graph.es['weight'] = weights
    # print graph.vs.select(['D1CF2F0F-A6FB-4768-A78F-06CA3097B271'])

    graph = nx.read_edgelist('./data.csv', delimiter='\t', data=(('weight', float),), encoding='utf-8')
    print('nodes=%s' % graph.number_of_nodes(), 'edges=%s' % graph.number_of_edges())
    # print graph.edges()
    partition = community.best_partition(graph)
    # print partition
    # print set(partition.values())
    res = pd.DataFrame(list(partition.iteritems()), columns=['id', 'type'])
    # print res
    print(res.groupby(['type']).count())
    res.to_csv('./result.csv', sep=',', encoding='utf-8', header=False, index=False)

# nx.draw(graph)
# plt.show()
# hive -e "select from_id,to_id from mining.dm_loan_usr_social_tmp_d">./data.csv
