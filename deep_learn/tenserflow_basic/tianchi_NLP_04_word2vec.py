# -*- coding: utf-8 -*-
"""
@Time     :2022/2/21 16:12
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import pandas as pd
import numpy as np

"""
天池NLP 一位同学的总结：
https://tianchi.aliyun.com/forum/postDetail?postId=122736
https://zhuanlan.zhihu.com/p/183862056?spm=5176.21852664.0.0.3bf33dd7roELXZ

感觉自己应该有这种总结的意识。
改进内容：
    1. 梳理整个流程，包含两大部分：数据处理（数据分布不均匀）和模型
    源代码不是从上到下顺序阅读的，更容易理解的做法为：先从整体上给出宏观的数据转换流程图，其中要包括数据在每一步的shape，以及包含的转换步骤，从而在心里构建一个框架图，再去看细节。

1. all_data2fold函数
    将原始的DataFrame数据，转换为一个list，    
"""



tianchi_data_path = "D:\Program Files\pythonwork\dataset\DeliberatePractice\deep_learn\\tianchi\\"
train_df = pd.read_csv(tianchi_data_path+"train_set.csv",
                       sep="\t",        # csv文件每列的分割符号 \t
                       # nrows=15000,
                       encoding="utf-8")       # 读取行数


def all_data2fold(fold_num, num=10000):
    """

    :param fold_num:
    :param num:
    :return:
    """
    flod_data = []
    texts = train_df['text'].tolist()[:num]
    labels = train_df['label'].tolist()[:num]

    total = len(labels)

    index = list(range(total))
    np.random.shuffle(index)

    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    # 构造一个dict key为label，value为list，存储对应类别包含的index
    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[id] = [i]
        else:
            label2id[label].append(i)
    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        batch_size = int(len(data)/fold_num)
        # other 表示多出来的数据，other数量小于fold_num。；最开始other是0
        other = len(data) - batch_size*fold_num
        #
        for i in range(fold_num):
            # 如果i < other 那么将一个数据添加到这一轮batch的数据中。
            cur_batch_size = batch_size + 1 if i < other else batch_size
            batch_data = [data[i*batch_size+b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data) # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。


