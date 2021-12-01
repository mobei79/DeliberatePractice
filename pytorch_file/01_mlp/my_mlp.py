# -*- coding: utf-8 -*-
"""
@Time     :2021/11/26 10:18
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc : NLP从入门到放弃的代码实例
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 自己实现一个Dataset类 去家在本地的csv数据集
# 需要三个函数 init get item len
class MyDataset(Dataset): # 继承pytorch自带的Dataset类
    def __init__(self, filepath): ## 加载原始数据集，并对特征数据和lable数据进行拆分
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index ):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = MyDataset('./sigmoid.csv')

# dataset是加载和拆分；传入DataLoader会进行：shuffle打乱数据，然后划分为mini-batch，用于深度学习模型；
# train_loader 是一个迭代器，每一个是batch—size大小的数据；
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# 定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6) # 全连接
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x)) # sigmoid 接入一个全连接的linear
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

#定义损失函数；二分类
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for i, data in enumerate(train_loader):
        inputs, labels = data   ## 获取数据
        y_pred = model(inputs)  ## 把数据喂进去给模型，获得结果
        loss = criterion(y_pred, labels)    ## 预测结果和真实值做损失函数
        print(epoch, i, loss.item())
        optimizer.zero_grad()   ## 反向传播之前要 梯度清零
        loss.backward()         ## 反向传播，更新参数