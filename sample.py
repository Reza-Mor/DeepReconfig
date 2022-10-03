#!/usr/bin/env python
# encoding: utf-8

import gym
import envs
import numpy as np

def run_one_episode (env, verbose=False):
    env.reset()
    sum_reward = 0

    #for i in range(env.MAX_STEPS): #should be max_steps
    for i in range(3):
        #action = env.action_space.sample()

        #pick a random action from uniform distribution (after masking)
        action = np.argmax(env.state["action_mask"] * np.random.uniform(low=0.0, high=1.0, size=len(env.state["action_mask"])))

        if verbose:
            print("action:", action)

        state, reward, done, info = env.step(action)
        sum_reward += reward

        if verbose:
            env.render()

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
    env = gym.make("rnaenv-v0")
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
