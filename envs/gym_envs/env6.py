import gym
from gym.utils import seeding
from gym.spaces import Space, Box, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import random
import math
import shelve

class Flows_v1 (gym.Env):

    def __init__ (self, dataset, max_episode_steps = np.inf):
        self.seed = 0
        self.dataset = dataset
        self.max_episode_steps = 60 #max_episode_steps
        self.set_seed()
        self.set_dataset_info() 
        # the action space is a set of discrete values (each node is a number)
        self.action_space = Discrete(self.num_flows)

        #self.observation_space = Dict({
        #    "adj_matrix": Box(low=0.0, high=self.max_capacity, shape=(self.n, self.n), dtype=np.float32),
        #    "curr_edge_capacities": Box(low=0.0, high=self.max_capacity, shape=(self.n, self.n), dtype=np.float32),
        #    "selected_flows": MultiBinary(self.num_flows),    
        #    "init_config": Box(low=0, high=max(self.num_flows, self.n), shape=((self.longest_flow_length + 2) * self.num_flows, ), dtype=np.float32),
        #    "target_config": Box(low=0, high=max(self.num_flows, self.n), shape=((self.longest_flow_length + 2) * self.num_flows, ), dtype=np.float32)
        #    })
        
        self.observation_space = Dict({
            "node_features": Box(low=-100, high=100, shape=(self.n, 3)),
            "edge_indices": Box(low=0, high=self.n, shape=(2, num_edges)),
            "adj_matrix": Box(low=0.0, high=self.max_capacity, shape=(self.n, self.n), dtype=np.float32),
            "curr_edge_capacities": Box(low=0.0, high=self.max_capacity, shape=(self.n, self.n), dtype=np.float32),
            "selected_flows": MultiBinary(self.num_flows),    
            "init_config": Box(low=0, high=max(self.num_flows, self.n), shape=((self.longest_flow_length + 2) * self.num_flows, ), dtype=np.float32),
            "target_config": Box(low=0, high=max(self.num_flows, self.n), shape=((self.longest_flow_length + 2) * self.num_flows, ), dtype=np.float32)
            })

        #print('self.observation_space: ', self.observation_space)

        self.reset()

    def reset (self):
        """
        Reset the state of the environment and returns an initial observation based on the 
        inputted graph (adjacancy matrix and node labels).

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.set_configurations()
        self.selected_flows = np.zeros(self.num_flows, dtype=np.int32)
        self.curr_edge_capacities = self.init_edge_capacities.copy()
        init_flows = self.convert_to_flat_array(self.init_config)
        target_flows = self.convert_to_flat_array(self.target_config)

        #print("init_flows: ", init_flows.shape)
        #print("init_flows: ", init_flows)
        
        # at first, we can only deselect the nodes on the LHS
        #action_mask = np.concatenate((np.ones(self.dim, dtype=np.int32), np.zeros(self.dim, dtype=np.int32)))
        
        self.init_state = {"adj_matrix": self.G,
                        "curr_edge_capacities": self.curr_edge_capacities,
                        "selected_flows": self.selected_flows,
                        "init_config": init_flows,
                        "target_config": target_flows
        }

        #print("self.curr_edge_capacities: ", type(self.curr_edge_capacities))
        #print("selected flows: ", type(self.selected_flows))
        #print("init_config: ", init_flows)
        #print("target_config: ", target_flows)
    
        self.count = 0
        self.state = self.init_state
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state


    def step (self, action):
        """
        The agent takes a step in the environment.

        input: actions representing a flow (each flow has an id)

        Returns observation, reward, done, info : tuple
        """
        if self.done:
            # code should never reach this point
            print("EPISODE DONE!!!")

        elif self.count == self.max_episode_steps:
            self.done = True

        else:
            assert self.action_space.contains(action)

            self.count += 1
            
            # get no reward unless an action is taken
            r = 0
            self.reward = r

            if self.selected_flows[action] == 0:
                prev_flow_path = self.init_config[action][:-1]
                target_flow_path = self.target_config[action][:-1]
                r = 1

            elif self.selected_flows[action] == 1:
                target_flow_path = self.init_config[action][:-1]
                prev_flow_path = self.target_config[action][:-1]
                r = -1

            flow_size = self.target_config[action][-1]
            can_move_flow = True
            i = 1
            target_edges = []

            # check if the flow can be moved to the target path
            while can_move_flow and i < len(target_flow_path) -1:
                target_edge = target_flow_path[i-1], target_flow_path[i]
                target_edges.append(target_edge)
                if flow_size > self.curr_edge_capacities[target_edge]:
                    can_move_flow = False
                i += 1

            # if possible, move the flow to the target position 
            if can_move_flow:
                for target_edge in target_edges:
                    self.curr_edge_capacities[target_edge] -= flow_size
                j = 1
                while j < len(prev_flow_path) -1:
                    prev_edge = prev_flow_path[j-1], prev_flow_path[j]
                    self.curr_edge_capacities[prev_edge] += flow_size
                    j += 1
                self.selected_flows[action] = 1 if self.selected_flows[action] == 0 else 0
                self.reward = r

            if np.all(self.selected_flows == 1):
                self.done = True

        return [self.state, self.reward, self.done, self.info]


    def set_dataset_info(self):
        db = shelve.open(self.dataset)
        self.G = db['G']
        print(type(self.G))
        self.n = self.G.shape[0]
        self.num_configs = db['num_configs']
        self.num_flows = db['num_flows']
        self.max_capacity = db['max_capacity']
        self.min_flow_size = db['min_flow_size']
        self.max_flow_size = db['max_flow_size']
        self.longest_flow_length = db['longest_flow_length']
        print('self.num_flows: ', self.num_flows)
        db.close()
    """
    
    """
    def set_configurations(self):
        db = shelve.open(self.dataset)
        i1, i2 = random.sample(range(self.num_configs), 2)
        self.init_config, self.init_edge_capacities = db['config_{}'.format(i1)]
        print('init_edge_c: ', self.init_edge_capacities)
        self.target_config, self.target_edge_capacities = db['config_{}'.format(i2)]
        db.close()

    def convert_to_flat_array(self, dict):
        print('flows: ', dict)
        lst = []
        for flow_id in dict.keys():
            lst.append(flow_id)
        for flow_path in dict.values():
            lst += flow_path
            zeros = [0] * (self.longest_flow_length - (len(flow_path)-1))
            lst += zeros
        return np.array(lst, dtype=np.float32)

    """
    node_features: N x node_dim matrix (N being the number of nodes)
    Return a F x node_dim matrix (F being the number of flows)
    """
    def get_flow_embeddings(self, node_features, flow_dict):
        node_indices= list(flow_dict.values())
        return np.sum(node_features[node_indices], axis =1)



    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max) 
        random.seed(seed)       
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

