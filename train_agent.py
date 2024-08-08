#!/usr/bin/env python
# encoding: utf-8

import gym
import envs
import numpy as np
from envs.gym_envs.env import Rnaenv_v1
from envs.gym_envs.env2 import Rnaenv_v2
from envs.gym_envs.env3 import Rnaenv_v3
from envs.gym_envs.env4 import Rnaenv_v4
from envs.gym_envs.env5 import Rnaenv_v5
from envs.gym_envs.env6 import Flows_v1
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

def main ():
    dataset = 'datasets/RNA/dataset1_20by20'
    env = Rnaenv_v2(dataset)

    # Instantiate the agent
    model = PPO("MultiInputPolicy", env, verbose=1)
    print(model.policy)
    # Train the agent and display a progress bar
    # model.learn(total_timesteps=int(2e5), progress_bar=True)
    model.learn(total_timesteps=10, progress_bar=True)

    print(model.policy)
    # Save the agent
    model.save("test_agent")

if __name__ == "__main__":
    main()
