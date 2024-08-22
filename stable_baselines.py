import os
from args import parse_args
# stable baselines 
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# environments
from envs.gym_envs.env import Rnaenv_v1
from envs.gym_envs.env2 import Rnaenv_v2
from envs.gym_envs.env3 import Rnaenv_v3
from envs.gym_envs.env4 import Rnaenv_v4
from envs.gym_envs.env5 import Rnaenv_v5
from envs.gym_envs.env6 import Flows_v1
from stable_baselines3.common.env_checker import check_env
# models 
from models.model_flows import GCN_Flows
from models.model_vc import GCN_VC
from torch_geometric.seed import seed_everything

def main(cfg):
    # Create environment
    env = get_env(cfg)
    #check_env(env)

    # Fix random seed
    seed_everything(cfg.seed)
    
    # Instantiate the agent
    model = get_model(cfg)

    if cfg.algorithm == 'PPO':
        policy_kwargs = dict(
            features_extractor_class=model,
            features_extractor_kwargs=dict(cfg= cfg), 
            net_arch=dict(pi=cfg.pi, vf=cfg.vf)
        )

        agent = PPO('MultiInputPolicy', env, learning_rate=cfg.lr, verbose=1, device=cfg.device, tensorboard_log=cfg.output_dir, policy_kwargs=policy_kwargs)
    
    elif cfg.algorithm == 'DQN':
        policy_kwargs = dict(
            features_extractor_class=model,
            features_extractor_kwargs=dict(cfg= cfg), 
            net_arch=cfg.vf
        )
        agent = DQN('MultiInputPolicy', env, learning_rate=cfg.lr, verbose=1, device=cfg.device, tensorboard_log=cfg.output_dir, policy_kwargs=policy_kwargs)

    # Train the agent
    agent.learn(total_timesteps=cfg.num_episodes, tb_log_name=cfg.project_name)
    
    # Save the agent
    ckpts_dir = os.path.join(cfg.output_dir, 'checkpoints')
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)
    save_path = os.path.join(ckpts_dir, cfg.project_name)
    agent.save(save_path)
    del agent  # delete trained model to demonstrate loading

    """
    # Load the trained agent
    agent = DQN.load("dqn_lunar")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(agent, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = agent.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()
    """

def get_model(cfg):
    if cfg.environment == "Rnaenv_v2":
        return GCN_VC
    elif cfg.environment == "Flows_v1":
        return GCN_Flows

def get_env(cfg):
    if cfg.environment == "Rnaenv_v2":
        return Rnaenv_v2(cfg = cfg)
    elif cfg.environment == "Flows_v1":
        return Flows_v1(cfg = cfg)

# max_reward = get_dataset_info(environment, dataset_path)
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


if __name__ == "__main__":
    
    cfg = parse_args()

    # write the run configs under the output dir
    with open('{}/{}'.format(cfg.output_dir, 'run_configs.txt'), 'w') as f:
        for key, value in cfg.__dict__.items():  
            f.write('%s:%s\n' % (key, value))
    f.close()

    main(cfg)
