# -*- coding: utf-8 -*-
"""
@Time     :2021/7/8 17:34
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from machine_learning.reinforcement_learning.Q_learning_demo_maze import Maze
from machine_learning.reinforcement_learning.Q_learning_demo_maze_RL_brain import QLearningTable


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
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()