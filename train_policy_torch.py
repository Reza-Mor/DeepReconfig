#!/usr/bin/env python
# encoding: utf-8

import gymnasium as gym
from envs.gym_envs.env import Rnaenv_v1
from envs.gym_envs.env2 import Rnaenv_v2
from envs.gym_envs.env3 import Rnaenv_v3
from envs.gym_envs.env4 import Rnaenv_v4
from envs.gym_envs.env5 import Rnaenv_v5
from envs.gym_envs.env6 import Flows_v1
from ray.tune.registry import register_env 
import os
import ray
#from ray import air, tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
#from ray.rllib.agents.ppo import PPOTrainer
#from ray.rllib.agents.ppo import PPOConfig
#from ray.rllib.agents.dqn import DQNTrainer
#from ray.rllib.agents.dqn import DQNConfig

#from ray.rllib.algorithms.dqn import DQNConfig
#from ray.rllib.algorithms.dqn import DQN

from ray import tune
from ray.rllib.models import ModelCatalog
import shutil
import shelve
import argparse
#from models import FeedForward, GAT, MODEL_CONFIG_1, MODEL_CONFIG_2, MODEL_CONFIG_3
#from models_pyG import GCN, TorchActionMaskModel
from models.model_flows import GCN_Flows
from models.model_vc import GCN_VC, TorchActionMaskModel
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()



dic = {'Rnaenv_v1': Rnaenv_v1, 'Rnaenv_v2': Rnaenv_v2, 'Rnaenv_v3': Rnaenv_v3, 
'Rnaenv_v4': Rnaenv_v4, 'Rnaenv_v5': Rnaenv_v5, 'Flows_v1': Flows_v1}

def get_dataset_info(environment, dataset_path):
    if environment in ['Rnaenv_v1', 'Rnaenv_v2', 'Rnaenv_v3', 'Rnaenv_v4', 'Rnaenv_v5']:
        db = shelve.open(dataset_path)
        dataset_size, max_graph_size, max_string_length =  db['non_zero_k'], db['max_graph_size'], db['max_string_length']
        max_reward =  max_graph_size * 2
        db.close()
        print("Training a model on a dataset of size {} of {}by{} graphs (RNA string length: {})".format(dataset_size, max_graph_size, max_graph_size, max_string_length))
    
    elif environment in ['Flows_v1']:
        db = shelve.open(dataset_path)
        num_flows = db['num_flows']
        print("Traning a model on a dataset with {} flows".format(num_flows))
        max_reward = num_flows

    return max_reward

def main(dataset, environment, model_config, n_iter, gamma):
    
    # init directory in which to save checkpoints
    ckpt_dir = "{}_{}_{}".format(dataset, environment, model_config)
    chkpt_root = "tmp/exa/{}".format(ckpt_dir)
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    # ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    # dataset_path = 'datasets/{}'.format(dataset)
    dataset_path = '{}'.format(dataset)
    ray_results = "results/ray_results/{}".format(dataset_path)
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # register the model for Rllib usage
    #if model_config == 'gat':
    #    ModelCatalog.register_custom_model("GAT", GAT)
    #else:
    #    ModelCatalog.register_custom_model("FeedForward", FeedForward)
    
    ModelCatalog.register_custom_model("GCN_VC", GCN_VC)
    ModelCatalog.register_custom_model("TorchActionMaskModel", TorchActionMaskModel)
    ModelCatalog.register_custom_model("GCN_Flows", GCN_Flows)

    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    #print(config)
    #config = dqn.DEFAULT_CONFIG.copy()

    #print(DQNConfig())
    #config = DQNConfig
    
    #config = PPOConfig()
    # register the custom environment
    select_env = environment
    register_env(select_env, lambda config: dic[environment](dataset_path))

    #config = dqn.DEFAULT_CONFIG.copy()
    config["num_workers"] = 0 #3
    config["disable_env_checking"]= True
    config["num_gpus"] = torch.cuda.device_count()
    config["gamma"] = gamma

    #config.rollouts(num_rollout_workers= 0)
    #config.resources(num_gpus= torch.cuda.device_count())


    MODEL_CONFIG_1 = {"custom_model": "GCN_Flows",
                       "custom_model_config": {"_disable_preprocessor_api": True}}         
    config["model"] = MODEL_CONFIG_1
    #config["model"]["no_final_linear"] = True
    config['framework'] = 'torch'
    
    #config.environment(select_env)
    config["log_level"] = "WARN"
    print(config["model"])
    #agent = ppo.PPOTrainer(config, env=select_env)
    agent = ppo.PPOTrainer(config=config, env=select_env)
    #algo = DQN(config=config, env=select_env)
    #algo.train()

    #Print the model summary
    #policy_network = agent.get_policy().model
    #policy_network.logits_and_value_model.summary() 

    max_reward = get_dataset_info(environment, dataset_path)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f}"
    n = 0

    while n < n_iter:
        result = agent.train()
        if n % 5 == 0:
            agent.save(chkpt_root)
        n += 1

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"]
                ))

        if result["episode_reward_mean"] > max_reward - 0.01:
            break

    # examine the trained policy
    #olicy = agent.get_policy()
    #model = policy.model.internal_model


   #tune.Tuner(
   # "PPO",
   # run_config=air.RunConfig(local_dir="results/ray_results/{}".format(dataset), name="{}_experiment".format(dataset), stop={"episode_reward_mean": 5},),
   # param_space={
   #     "env": "rnaenv-v0",
   #     "num_gpus": len(tf.config.list_physical_devices('GPU')),
   #     "num_workers": 1,
   #     "lr": tune.grid_search([0.0005, 0.0001]),
   #     "model": tune.grid_search([MODEL_CONFIG_1, MODEL_CONFIG_2])
   # },
   # ).fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        #default='RNA/dataset1_20by20',
        default='datasets/flows/dataset_2',
        help="the file to load the dataset from",
    )
    parser.add_argument(
        "--environment",
        type=str,
        #default='Rnaenv_v2',
        default='Flows_v1',
        help="the environment to train the model on",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default='ff1',
        help="the model configurations as set in models.py. model_config must be one of ff1, ff2, gat",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="number of iterations",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor",
    )
    
    args = parser.parse_args()
    if args.dataset == None:
        print('Must specify a file to read from')
    if not args.environment in dic:
        print('Environment must be defined- check the env dic')
    #if not args.model_config in model_configs:
    #    print('Model_config must be specified in model_configs')
    if args.gamma > 1 or args.gamma < 0:
        print('gamma must be between 0 and 1')
    else:
        main(args.dataset, args.environment, args.model_config, args.n_iter, args.gamma)
