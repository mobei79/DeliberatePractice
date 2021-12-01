# -*- coding: utf-8 -*-
"""
@Time     :2021/11/29 17:00
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc : 一篇博客的代码： https://blog.csdn.net/qq_42255269/article/details/112734754
"""
import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import os

print(torch.__version__)
# 处理数据集
# PTB（Penn Tree Bank）是一个常用的小型语料库 。它采样自《华尔街日报》的文章，包括训练集、验证集和测试集。我们将在PTB训练集上训练词嵌入模型。
# 该数据集的每一行作为一个句子。句子中的每个词由空格隔开。
# 下载链接 http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

# print(os.listdir("../ref/data/ptb"))
print(os.listdir("../word2vec/simple-examples/data"))
assert 'ptb.train.txt' in os.listdir("../word2vec/simple-examples/data")


with open('../word2vec/simple-examples/data/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

'# sentences: %d' % len(raw_dataset) # 输出 '# sentences: 42068'

# 对于数据集的前5个句子，打印每个句子的词数和前5个词。这个数据集中句尾符为""，生僻词全用"“表示，数字则被替换成了"N”。

for st in raw_dataset[:5]:
    print('# tokens:', len(st), st[:5])

# 建立词语索引
# 只保留在数据集中至少出现5次的词。

# tk是token的缩写
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

# 然后将词映射到整数索引。

idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
'# tokens: %d' % num_tokens # 输出 '# tokens: 887100'

# 二次采样
# 文本数据中一般会出现一些高频词，如英文中的“the”“a”和“in”。通常来说，在一个背景窗口中，一个词（如“chip”）和较低频词（如“microprocessor”）
# 同时出现比和较高频词（如“the”）同时出现对训练词嵌入模型更有益。因此，训练词嵌入模型时可以对词进行二次采样 [2]。
# 具体来说，数据集中每个被索引词wi将有一定概率被丢弃，该丢弃概率为
# https://blog.csdn.net/qq_42255269/article/details/112734754?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.essearch_pc_relevant&spm=1001.2101.3001.4242.1

def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens) #概率丢弃

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
'# tokens: %d' % sum([len(st) for st in subsampled_dataset]) # '# tokens: 375875'

# 二次采样后去掉了一半左右的词。下面比较一个词在二次采样前后出现在数据集中的次数。可见高频词“the”的采样率不足1/20。

def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

compare_counts('the') # '# the: before=50770, after=2013'

# 低频词“join”则完整地保留了下来。

compare_counts('join') # '# join: before=45, after=45'

# 提取中心词和背景词
# 我们将与中心词距离不超过" 背景窗口 "大小的词作为它的背景词。下面定义函数提取出所有中心词和它们的背景词。
# 它每次在整数1和max_window_size（最大背景窗口）之间随机均匀采样一个整数作为背景窗口大小。
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

# 创建一个人工数据集，其中含有词数分别为7和3的两个句子。设最大背景窗口为2，打印所有中心词和它们的背景词。

tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)

# 设最大背景窗口大小为5。提取数据集中所有的中心词及其背景词。
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

# 使用负采样来进行近似训练。对于一对中心词和背景词，我们随机采样K个噪声词（实验中设K = 5 ）。
# 根据word2vec论文的建议，噪声词采样概率P(w)设为w 词频与总词频之比的0.75次方
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K: #K negative for one background
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5)) #参数weights设置相对权重，它的值是一个列表，设置之后，每一个成员被抽取到的概率就被确定了。
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg) #select negative from 10**5 candidates
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

# 这里展示一下K与背景词数量、干扰词数量的关系：
K = 5
print(" {} * len of ({}) = len of({})".format(K, all_contexts[0], all_negatives[0]))

"""
读取数据
从数据集中提取所有中心词all_centers，以及每个中心词对应的背景词all_contexts和噪声词all_negatives。
我们先定义一个Dataset类。
"""
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)#trigger for exception
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)

"""
通过随机小批量来读取它们。在一个小批量数据中，第i个样本包括一个中心词以及它所对应的n_i个背景词和m_i个噪声词。
    由于每个样本的背景窗口大小可能不一样，其中背景词与噪声词个数之和n_i + m_i也会不同。
    在构造小批量时，我们将每个样本的背景词和噪声词连结在一起，并添加填充项0直至连结后的长度相同，即长度均为max x_i * n_i + m_i （max_len变量）。
    为了避免填充项对损失函数计算的影响，我们构造了掩码变量masks，其每一个元素分别与连结后的背景词和噪声词contexts_negatives中的元素一一对应。
    当contexts_negatives变量中的某个元素为填充项时，相同位置的掩码变量masks中的元素取0，否则取1。
    为了区分正类和负类，我们还需要将contexts_negatives变量中的背景词和噪声词区分开来。
    依据掩码变量的构造思路，我们只需创建与contexts_negatives变量形状相同的标签变量labels，并将与背景词（正类）对应的元素设1，其余清0。
下面我们实现这个小批量读取函数batchify。它的小批量输入data是一个长度为批量大小的列表，其中每个元素分别包含中心词center、背景词context和噪声词negative。
该函数返回的小批量数据符合我们需要的格式，例如，包含了掩码变量。

"""
def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list,
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))
# 用刚刚定义的batchify函数指定DataLoader实例中小批量的读取方式，然后打印读取的第一个批量中各个变量的形状。
batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4

dataset = MyDataset(all_centers,
                    all_contexts,
                    all_negatives)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True, #set a dataset for
                            collate_fn=batchify,
                            num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break
#输出
# centers shape: torch.Size([512, 1])
# contexts_negatives shape: torch.Size([512, 60])
# masks shape: torch.Size([512, 60])
# labels shape: torch.Size([512, 60])

"""
跳字模型
通过使用嵌入层和小批量乘法来实现跳字模型。它们也常常用于实现其他自然语言处理的应用。
嵌入层
    获取词嵌入的层称为嵌入层，在PyTorch中可以通过创建nn.Embedding实例得到。嵌入层的权重是一个矩阵，其行数为词典大小（num_embeddings），
    列数为每个词向量的维度（embedding_dim）。我们设词典大小为20，词向量的维度为4。
嵌入层的输入为词的索引。输入一个词的索引i ii，嵌入层返回权重矩阵的第i ii行作为它的词向量。下面我们将形状为(2, 3)的索引输入进嵌入层，由于词向量的维度为4，我们得到形状为(2, 3, 4)的词向量。

"""
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
embed.weight

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
embed(x)

"""
小批量乘法
可以使用小批量乘法运算bmm对两个小批量中的矩阵一一做乘法。
"""
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
torch.bmm(X, Y).shape

"""
跳字模型前向计算
"""
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)#(batch_size, 1)
    u = embed_u(contexts_and_negatives)#(batch_size, max_len)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
"""
训练模型
定义二元交叉熵损失函数
"""
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)

loss = SigmoidBinaryCrossEntropyLoss()

pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签变量label中的1和0分别代表背景词和噪声词
label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
loss(pred, label, mask) * mask.shape[1] / mask.float().sum(dim=1)


def sigmd(x):
    return - math.log(1 / (1 + math.exp(-x)))

print('%.4f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4)) # 注意1-sigmoid(x) = sigmoid(-x)
print('%.4f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))


"""
初始化模型参数
分别构造中心词和背景词的嵌入层，并将超参数词向量维度embed_size设置成100。
"""
embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
)

# 定义训练函数
# 由于填充项的存在，与之前的训练函数相比，损失函数的计算稍有不同
def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])

            # 使用掩码变量mask来避免填充项对损失函数计算的影响
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean() # 一个batch的平均loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))

train(net, 0.01, 10)


"""
应用词嵌入模型
根据两个词向量的余弦相似度表示词与词之间在语义上的相似度
"""
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt() #词向量之间的关联性
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))

get_similar_tokens('news', 10, net[0])
