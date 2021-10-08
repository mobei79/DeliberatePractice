# -*- coding: utf-8 -*-
"""
@Time     :2021/7/8 17:34
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from machine_learning.reinforcement_learning.Q_learn.Q_learning_demo_maze import Maze   # 环境
from machine_learning.reinforcement_learning.Q_learn.Q_learning_demo_maze_RL_brain import QLearningTable    # agent 大脑


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                break

if __name__ == "__main__":

    # 将环境变化和环境奖励放到environment中处理；将动作选择，更新q表放到agent中

    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()