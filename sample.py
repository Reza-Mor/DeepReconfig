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

def run_one_episode (env, verbose=False):
    env.reset()
    sum_reward = 0
    
    #for i in range(env.MAX_STEPS): #should be max_steps
    for i in range(10):
        #action = env.action_space.sample()

        #pick a random action from uniform distribution (after masking)
        #action = np.argmax(env.state["action_mask"] * np.random.uniform(low=0.0, high=1.0, size=len(env.state["action_mask"])))
        action = np.argmax(np.random.uniform(low=0.0, high=1.0, size=len(env.state["selected_flows"])))

        if verbose:
            print("action:", action)

        state, reward, done, info = env.step(action)
        sum_reward += reward

        if verbose:
            print('state: ', state)
            print('reward: ', reward)
            #env.render()
            #print('diff: ', abs(state['adj_matrix']- prev_state))
            #prev_state = state['adj_matrix']

        if done:
            if verbose:
                print("done @ step {}".format(i))

            break

    if verbose:
        print("cumulative reward", sum_reward)

    return sum_reward


def main ():
    # first, create the custom environment and run it for one episode
    #env = gym.make("example-v0")
    #kwargs = {'dataset':'datasets.expert_dbCRW_AND_entry_typeSequence_bonds_5by5'}
    #env = gym.make("rnaenv-v0")
    dataset = 'datasets/flows/dataset_1'
    #env = Rnaenv_v5(dataset)
    env = Flows_v1(dataset)
    sum_reward = run_one_episode(env, verbose=True)

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    #for _ in range(10):
    #    sum_reward = run_one_episode(env, verbose=False)
    #    history.append(sum_reward)

    #avg_sum_reward = sum(history) / len(history)
    #print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))


if __name__ == "__main__":
    main()
