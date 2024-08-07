import gym
from gym.utils import seeding
from gym.spaces import Space, Box, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import random
import math
import shelve
from torch_geometric.utils import dense_to_sparse
import torch

class Flows_v1 (gym.Env):

    def __init__ (self, dataset, max_episode_steps = np.inf):
        self.seed = 0
        self.dataset = dataset
        self.max_episode_steps = 60 #max_episode_steps
        self.set_seed()
        self.set_observation_space() 
        # the action space is a set of discrete values (each node is a number)
        self.action_space = Discrete(self.num_flows)

        self.reset()

    def set_observation_space(self):
        db = shelve.open(self.dataset)
        self.G = db['G'] # the adj matrix
        print('G: ', dense_to_sparse(torch.from_numpy(self.G)))
        self.n = self.G.shape[0]
        torch.manual_seed(1)
        dense_G = dense_to_sparse(torch.from_numpy(self.G))
        self.edge_indices, self.edge_capacities = dense_G[0], dense_G[1] #coo adj matrix
        self.features = torch.rand(self.n, 3) # set node features
        self.num_configs = db['num_configs']
        self.num_flows = db['num_flows']
        self.max_capacity = db['max_capacity']
        self.min_flow_size = db['min_flow_size']
        self.max_flow_size = db['max_flow_size']
        self.longest_flow_length = db['longest_flow_length']
        num_edges = len(self.edge_indices[0])
        db.close()
    
        self.observation_space = Dict({
            "node_features": Box(low=-100, high=100, shape=(self.n, 3)),
            "edge_indices": Box(low=0, high=self.n, shape=(2, num_edges)),
            "edge_features": Box(low=0, high=self.n, shape=(num_edges, 3)),
            #"edge_capacities": Box(low=0.0, high=self.max_capacity, shape=(self.n, self.n), dtype=np.float32),
            #"curr_edge_capacities": Box(low=0.0, high=self.max_capacity, shape=(self.n, self.n), dtype=np.float32),
            "selected_flows": MultiBinary(self.num_flows),    
            "init_config": Box(low=0, high=self.n, shape=((self.longest_flow_length + 1) * self.num_flows, ), dtype=np.float32),
            "target_config": Box(low=0, high=self.n, shape=((self.longest_flow_length + 1) * self.num_flows, ), dtype=np.float32),
            "indices_init": Box(low=0, high=self.n, shape=(self.num_flows, ), dtype=np.float32),
            "indices_target": Box(low=0, high=self.n, shape=(self.num_flows, ), dtype=np.float32),
            "flow_sizes": Box(low=0, high=self.n, shape=(self.num_flows, ), dtype=np.float32)
            })
        
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
        init_flows, flow_sizes, indices_init = self.convert_to_flat_array(self.init_config)
        target_flows, _, indices_target = self.convert_to_flat_array(self.target_config)
        self.update_edge_features(self.edge_capacities, self.curr_edge_capacities)

        self.init_state = {"node_features": self.features.numpy(),
                           "edge_indices": self.edge_indices.numpy(),
                           "edge_features": self.edge_features,
                           #"edge_capacities": self.edge_capacities,
                           #"curr_edge_capacities": self.curr_edge_capacities,
                           "selected_flows": self.selected_flows,
                           "init_config": init_flows,
                           "target_config": target_flows,
                           "indices_init": indices_init,
                           "indices_target": indices_target,
                           "flow_sizes": flow_sizes,
                           
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

        self.update_edge_features(self.edge_capacities, self.curr_edge_capacities)

        return [self.state, self.reward, self.done, self.info]

    def set_configurations(self):
        db = shelve.open(self.dataset)
        i1, i2 = random.sample(range(self.num_configs), 2)
        self.init_config, self.init_edge_capacities = db['config_{}'.format(i1)]
        self.target_config, self.target_edge_capacities = db['config_{}'.format(i2)]
        #print('init_config: ', self.init_config)
        #print('target_config: ', self.target_config)
        db.close()

    def convert_to_flat_array2(self, dict):
        #print('flows: ', dict)
        lst = []
        for flow_id in dict.keys():
            lst.append(flow_id)
        for flow_path in dict.values():
            lst += flow_path
            zeros = [0] * (self.longest_flow_length - (len(flow_path)-1))
            lst += zeros
        return np.array(lst, dtype=np.float32)
    
    def convert_to_flat_array(self, dict):
        flow_paths = []
        flow_sizes = []
        indices = []
        total_index= 0
        for flow_id, flow in sorted(dict.items()):
            flow_paths += flow[:-1]
            total_index += len(flow[:-1])
            indices.append(total_index)
            flow_sizes.append(flow[-1])

        #print('dict1: ', dict)
        #print(np.split(np.array(flow_paths), indices[:-1]))
        return np.array(flow_paths, dtype=np.float32), np.array(flow_sizes, dtype=np.float32), indices[:-1]
    
    """
    edge_capacities: array of size E with edge capacities
    curr_edge_capacities: N by N array with current edge capacities

    Returns a E x 3 matrix of edge features.
    E[0-th column]: the total edge capacity
    E[1-st column]: current free edge capacity 
    E[2-nd column]: current used edge capacity 
    """
    def update_edge_features(self, edge_capacities, curr_edge_capacities):
        
        # the total edge capacities added to the current edge capacities, only used for processing the data 
        sparse_curr_edge_cap_total = dense_to_sparse(torch.from_numpy(self.G + curr_edge_capacities))[1]

        sparse_curr_edge_cap = sparse_curr_edge_cap_total - edge_capacities
        sparse_curr_edge_used = edge_capacities - sparse_curr_edge_cap
        
        self.edge_features = torch.cat((torch.unsqueeze(edge_capacities,1),
                             torch.unsqueeze(sparse_curr_edge_cap,1),
                             torch.unsqueeze(sparse_curr_edge_used,1)), 1)

    """
    Processes the input data to obtain a list of flow paths
    Example: converts [1, 2, 3, 1, 4, 5, 3] with indices 3 to [[1,2,3], [1, 4, 5, 3]].
    [[1,2,3], [1, 4, 5, 3]] means flow 0 goes through nodes 1,2,3 and flow 1 goes through nodes 1, 4, 5, 3
    """
    def get_flow_paths_as_list(flow_lst, indices):
        flow_paths = np.split(np.array(flow_lst), indices)
        # change data type
        return list(map(list, flow_paths))
        
    """
    node_features: N x node_dim matrix (N being the number of nodes)
    Return a F x node_dim matrix (F being the number of flows)
    Each flow embedding is the sum of the node embeddings
    """
    def get_flow_embeddings(self, node_features, flow_paths):
        return np.sum(node_features[flow_paths], axis =1)

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max) 
        random.seed(seed)       
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

