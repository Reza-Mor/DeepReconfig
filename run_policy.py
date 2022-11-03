#!/usr/bin/env python
# encoding: utf-8

from envs.gym_envs.env import Rnaenv_v0
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
from ray.rllib.models import ModelCatalog
import shutil
from models import ActionMaskModel, MODEL_CONFIG_1
import json

def main ():

    # register the model for Rllib usage
    ModelCatalog.register_custom_model("CustomModel", ActionMaskModel)
    
    # register the custom environment
    select_env = "rnaenv-v0"
    dataset = 'expert_dbCRW_AND_entry_typeSequence_10by10'
    dataset_path = 'datasets/{}'.format(dataset)
    register_env(select_env, lambda config: Rnaenv_v0(dataset_path))

    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    #config = dqn.DEFAULT_CONFIG.copy()
    config["model"] = MODEL_CONFIG_1
   
    agent = ppo.PPOTrainer(config, env=select_env)
    env = Rnaenv_v0(dataset_path)
    
    checkpoint_indexes = [i for i in range(1,99,5)]
    #chkpoints = ["tmp/exa/checkpoint_000030"]

    state = env.reset()
    sum_reward = 0
    num_episodes = 1
    n_step = env.max_episode_steps
    print_info = False
    Actions = []
    R_selected = []
    L_selected = []
    Energy = []
    dictionary = {}

    output_file = 'results/{}.json'.format(dataset)
    f = open(output_file, 'a')

    for chkpt_indx in checkpoint_indexes:
        i = str(chkpt_indx) if chkpt_indx > 9 else "0" + str(chkpt_indx)
        chkpt = "tmp/exa/{}_easy/checkpoint_0000{}".format(dataset, i)
        agent.restore(chkpt)
        # apply the trained policy in a rollout
        for _ in range(num_episodes):
            for _ in range(n_step):
                #action = agent.compute_action(state)
                action = agent.compute_single_action(state)
                state, reward, done, info = env.step(action)
                sum_reward += reward
                
                if print_info:
                    #env.render()
                    print('Action: {},  Reward: {}'.format(action, reward))

                Actions.append(int(action))
                R_selected.append(int(info["num_right_selected"]))
                L_selected.append(int(info["num_left_selected"]))
                Energy.append(int(info["curr_energy"]))

                if done == 1:
                    # report at the end of each episode
                    print("chkpt: ", chkpt)
                    print("Cumulative Reward", sum_reward)
                    print("Actions: ", Actions)

                    state = env.reset()
                    sum_reward = 0
                    dictionary['Dataset'] = dataset
                    dictionary['chkpt'] = chkpt_indx
                    dictionary['Actions'] = Actions
                    dictionary['Energy'] = Energy
                    dictionary['L_selected'] = L_selected
                    dictionary['R_selected'] = R_selected
                    json.dump(dictionary, f)
                    f.write('\n')
                    Actions = []
                    R_selected = []
                    L_selected = []
                    Energy = []
                    dictionary = {}
                    break

    f.close()

if __name__ == "__main__":
    main()
