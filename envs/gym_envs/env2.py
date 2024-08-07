#import gymnasium as gym
import gym
from gym.utils import seeding
from gym.spaces import Box, Dict, Discrete, MultiBinary#, Graph
import numpy as np
import networkx as nx
import math
from networkx.algorithms import bipartite
from torch_geometric.utils import dense_to_sparse, from_networkx
import shelve
import torch

class Rnaenv_v2 (gym.Env):

    def __init__ (self, dataset, max_episode_steps = np.inf):
        self.seed = 0
        self.dataset = dataset
        self.max_episode_steps = 100 #max_episode_steps
        self.set_seed()
        self.set_n()
        #self.n = 10
        self.features = torch.rand(self.n, 3) #3 is the feature dimension
        # the action space is a set of discrete values (each node is a number)
        self.action_space = Discrete(self.n//2)
        self.set_observation_space()

        self.reset()

    def set_observation_space(self):
        self.dim = self.n//2
        self.biadj_matrix, self.G = self.get_graph()
        num_edges = self.G.edge_index.shape[1]

        self.observation_space = Dict({
            #"graph": Graph(node_space=Box(low=-100, high=100, shape=(self.n, 2)), edge_space=Discrete(3)),
            "node_features": Box(low=-100, high=100, shape=(self.n, 3)),
            "edge_indices": Box(low=0, high=self.n, shape=(2, num_edges)), #Discrete(3),
            "selected_left_nodes": MultiBinary(self.n//2),
            "selected_right_nodes": MultiBinary(self.n//2),
            "action_mask": MultiBinary(self.n//2),
            "energy_dist": Box(-10, self.n, shape=(2,), dtype=np.int32)
        })

    def reset (self):
        """
        Reset the state of the environment and returns an initial observation based on the 
        inputted graph (adjacancy matrix and node labels).

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.set_observation_space()

        # need to fix this
        # self.biadj_matrix, self.coo_indices = self.get_graph_test()
        selected_left_nodes = np.ones(self.dim, dtype=np.int32)
        selected_right_nodes = np.zeros(self.dim, dtype=np.int32)
        action_mask = np.ones(self.dim, dtype=np.int32)
        # at first, we can only deselect the nodes on the LHS
        self.curr_energy = self.dim #the current energy is simply the number of selected nodes
        
        # To do: encode information if node is on left\right side

        self.init_state = {
            "node_features": self.features.numpy(),
            "edge_indices": self.G.edge_index.numpy(),
            "selected_left_nodes": selected_left_nodes,
            "selected_right_nodes": selected_right_nodes,
            "action_mask": action_mask,
            "energy_dist": np.array([self.k, self.curr_energy - self.k])
        }

        self.goal = np.ones(self.dim, dtype=np.int32)       
        self.count = 0
        self.state = self.init_state
        self.reward = 0
        self.max_selected_right = 0
        self.selected_indexes_prev = None #this is used for energy level implementation
        #self.selected = None #True if the current action is selecting a node
        self.done = False
        self.info = {}

        return self.state


    def step (self, action):
        """
        The agent takes a step in the environment.

        input: actions representing a node (each node has a number)

        Returns observation, reward, done, info : tuple
        """
        if self.done:
            # code should never reach this point
            print("EPISODE DONE!!!")

        elif self.count == self.max_episode_steps:
            self.done = True

        else:
            #assert self.action_space.contains(action)

            self.count += 1

            # update the state
            self.update_state(action)

        #print('env state: ', self.init_state)

        return [self.state, self.reward, self.done, self.info]


    def update_state(self, action):

        #Ture if node is being deselected, False if being selected
        selected = (self.state["selected_right_nodes"][action] == 1)

        # find the nieghbours of the new selected\deselected node 
        ngbrs_lst = self.biadj_matrix[:, action]
        ngbrs_index = np.where(ngbrs_lst == 1)[0]

        # self.curr_energy - self.k >= num_prev_selected_left_nodes checks if the threshold will not go under k after the selection
        prev_selected_left_nodes = self.state["selected_left_nodes"] * ngbrs_lst # ngbrs_lst.T
        num_prev_selected_left_nodes = sum(prev_selected_left_nodes)

        if not selected and self.curr_energy - self.k >= num_prev_selected_left_nodes:
            self.curr_energy -= (num_prev_selected_left_nodes - 1)
            self.state["selected_left_nodes"][ngbrs_index] = 0
            self.state["selected_right_nodes"][action] = 1
            self.reward = 1

        elif selected:

            # update the state
            self.state["selected_right_nodes"][action] = 0

            if ngbrs_index.shape != (0,):
                # ngbrs_rows: each row is a neighbour n of the RHS node selected
                # each cloumn is the neighbours of the node n |neighbours| by |RHS| matrix
                ngbrs_rows = self.biadj_matrix[ngbrs_index, :]
                # m[i,j] is 1 iff the j-th node in RHS is selected and is a neighnbour of node i in LHS 
                m = np.multiply(ngbrs_rows, self.state["selected_right_nodes"])
                # mask[i] is true if node i in LHS can be selected (all the neighbours of the i-th node in RHS are not selected)
                mask = abs(1 - np.any(m, axis = 1))
                # indexes of neighbour nodes in LHS which have no selected neighbour in RHS
                indexs = np.multiply(ngbrs_index, mask)
                self.state["selected_left_nodes"][indexs] = 1
                self.curr_energy += (mask.sum() - 1)
            self.reward = -1
        
        else:
            self.reward = 0


        if np.array_equal(self.goal, self.state["selected_right_nodes"]):
            self.done = True
            self.reward += self.dim
            
        #try:
        #    assert self.observation_space.contains(self.state)
        #except AssertionError:
        #    print("INVALID STATE", self.state)

        # make the "energy unavailable nodes (deselected nodes due to low energy)" available if the energy is above threshold
        if self.curr_energy >= self.k and type(self.selected_indexes_prev) != type(None):
            self.state["action_mask"][self.selected_indexes_prev] = 1

        # if energy < k, cannot deselect any selected nodes
        elif self.curr_energy < self.k:
            selected_indexes = np.where(self.state["selected_right_nodes"] == 1)[0]
            self.selected_indexes_prev =  selected_indexes.copy()
            self.state["action_mask"][selected_indexes] = 0
            #print('hit low energy')
        
        #a= np.array([self.k, self.curr_energy - self.k], dtype=np.int32)
        #torch.from_numpy(a)

        self.state["energy_dist"] = np.array([self.k, self.curr_energy - self.k], dtype=np.int32)

        # distance is the number of RHS nodes that are not selected
        self.info["curr_energy"] = self.curr_energy
        self.info["num_right_selected"] = self.state["selected_right_nodes"].sum()
        self.info["num_left_selected"] = self.state["selected_left_nodes"].sum()

    """
    Loads a random graph from the dataset in form of an adj matrix and sets the threshold k 
    """
    def get_graph(self):
        db = shelve.open(self.dataset)
        dataset_size = db['dataset_size'] 
        graph_size = db['max_graph_size']
        x = np.random.randint(0, dataset_size)
        x = 10
        dic = db[str(x)]

        while True:
            G = dic['graph']
            u, v = dic['size']
            geometric_data = from_networkx(G)
            # set the node features
            geometric_data.x = self.features
            #coo_indices = geometric_data.edge_index
            biadj_matrix = bipartite.biadjacency_matrix(G, np.array(range(u)), u + np.array(range(v))).todense()
            # pad the matrix with zeros for having the same shape across all instances
            if u < graph_size:
                biadj_matrix = np.concatenate((biadj_matrix, np.zeros((graph_size - u, v))), axis=0)
                u = graph_size
            if v < graph_size:
                biadj_matrix = np.concatenate((biadj_matrix, np.zeros((u, graph_size - v))), axis=1)

            #if dic['k'] == None:
                #arbitrary placeholder
                #self.k = u//2 
            if dic['k'] != 0:
                #self.k = dic['k']
                self.k =  dic['k'] #u//2 
                break
        db.close()
        self.k = 10
       
        return biadj_matrix, geometric_data


    def get_graph_test(self):
        biadj_matrix = np.array([[1., 0., 0., 0., 0.],
                                [1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [1., 1., 1., 1., 0.],
                                [1., 1., 1., 1., 1.]])
        self.k =  4
        return biadj_matrix

    def set_n(self):
        """
        Get the number of nodes (max graph size * 2) in the dataset and set it to n
        """
        db = shelve.open(self.dataset)
        self.n = 2 * db['max_graph_size']
        db.close()


    def render (self, mode="graph"):
        """Renders the environment.
        Args:
            mode (str): the mode to render with 
            graph gives the graph
            "print" gives the state
        """
        #UPDATE
        print(self.state)


    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)        
        self.np_random, seed = seeding.np_random(seed)
        torch.manual_seed(seed)
        return [seed]

