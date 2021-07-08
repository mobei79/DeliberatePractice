# -*- coding: utf-8 -*-
"""
@Time     :2021/7/8 13:03
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import pandas as pd
import numpy as np
import time

"""
包含如下步骤：
    创建Q表
    创建环境（编写复杂）
    选择动作的功能
    更新步骤
"""

# 初始化
np.random.seed(2) # 函数用于生成指定随机数。
# global variable
N_STATES = 6 # 一维世界的长度
ACTIONS = ['left','right'] # 所有的动作
EPSILON = 0.9 # 选择策略
ALPHA = 0.1 # 学习效率
LAMBDA = 0.9 # 衰减参数
MAX_EPISODES = 13 # maximum episodes 玩13回合就训练好了
FRESH_TIME = 0.3 # fresh time for one move

# 创建Q表 【包含每个每个状态，每个动作的价值】
def buile_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns= actions,
    )
    # print(table)
    return table

# 动作选择功能 【根据当前status和Q表选择】
def choose_action(states, q_table):
    state_actions = q_table.iloc[states,:]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0) : # 10%的记录随机选择
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

# 环境的反映 环境会对我们的行为做出什么反映（feedback）: 当前S采取行为A下一步会到达那个状态S_，以及这个行为获得什么奖励R
# 其实包含很多步：奖励机制；环境状态变换；
def get_env_feedback(S, A):
    # this is how agent will interact with the environment agent与环境交互的方式
    S_ = 0
    if A == "right":
        if S == N_STATES -2:
            S_ = "terminal"
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

# 创建环境 探索者如何移动
def update_env(S, episode, step_count):
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == "terminal":
        interaction = "Episode %s: total_steps = %s"%(episode+1, step_count)
        print('{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                                                 ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

# 主循环
def rl():
    q_table = buile_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_count = 0
        S = 0 # 初识状态为0
        is_terminal = False # 本轮终止符
        update_env(S, episode, step_count)
        while not is_terminal:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            # print("*"*19)
            # print("当前状态：%d,下一个动作%d,选择的动作%s"%(S,S_,A))
            q_predict = q_table.at[S, A] # 估计值 也就是原来的值
            if S_ != 'terminal':
                q_target = R + LAMBDA*q_table.iloc[S_,:].max() # 真实值
            else:
                q_target = R
                is_terminal = True
            q_table.at[S,A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_count+1)
            step_count += 1
    return q_table #

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

# buile_q_table(N_STATES, ACTIONS)