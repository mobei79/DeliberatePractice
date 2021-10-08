# -*- coding: utf-8 -*-
"""
@Time     :2021/9/3 9:40
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

from maze_env import Maze
from RL_brain import DeepQNetWork

def run_maze():
    step = 0
    for episode in range(300):
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # RLcho


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetWork(env.n_action, env.n_features)
