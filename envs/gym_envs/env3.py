import gym
from gym.utils import seeding
from gym.spaces import Space, Box, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import math
from networkx.algorithms import bipartite
import shelve

class Rnaenv_v3 (gym.Env):

    def __init__ (self, dataset, max_episode_steps = np.inf):
        self.seed = 0
        self.dataset = dataset
        self.max_episode_steps = 100 #max_episode_steps
        self.set_seed()
        self.set_n()
        #self.n = 10
        # the action space is a set of discrete values (each node is a number)
        self.action_space = Discrete(self.n)

        # NOTE: the actual state does not contain the action mask
        self.observation_space = Dict({
            "adj_matrix": MultiBinary([self.n//2, self.n//2]), 
            "selected_nodes": MultiBinary(self.n),
            "action_mask": Box(0, 1, shape=(self.n,), dtype=np.int32),
            "energy_dist": Box(-1, self.n, shape=(2,), dtype=np.int32)
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
        self.dim = self.n//2
        #adj_matrix = self.get_graph()
        adj_matrix = self.get_graph()
        selected_nodes = np.concatenate((np.ones(self.dim, dtype=np.int32), np.zeros(self.dim, dtype=np.int32)))
        # at first, we can only deselect the nodes on the LHS
        action_mask = np.concatenate((np.ones(self.dim, dtype=np.int32), np.zeros(self.dim, dtype=np.int32)))
        self.curr_energy = self.dim #the current energy is simply the number of selected nodes
        self.init_state = {"adj_matrix": adj_matrix,
                        "selected_nodes": selected_nodes,
                        "action_mask": action_mask,
                        "energy_dist": np.array([self.k, self.curr_energy - self.k], dtype= np.int32)}

        self.goal = np.concatenate((np.zeros(self.dim),np.ones(self.dim)))         
        self.count = 0
        self.state = self.init_state
        self.reward = 0
        self.right_selected = 0
        self.left_selected = self.dim
        self.max_selected_right = 0
        self.selected_indexes_prev = None #this is used for energy level implementation
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
            assert self.action_space.contains(action)

            self.count += 1

            # R is defined as the number of nodes in right side that are selected
            #R = (self.state["selected_nodes"][self.dim:]).sum()
            
            #reward = max(R - self.max_selected_right, 0)
            #reward = 0
            #self.max_selected_right =  R if R > self.max_selected_right else self.max_selected_right

            if action >= self.dim and self.state["selected_nodes"][action] == 0: # if select a node in the right side
                reward = 1
                self.right_selected += 1

            elif action >= self.dim and self.state["selected_nodes"][action] == 1: #if deselect a node in the right side
                reward = -1
                self.right_selected-= 1

            elif action < self.dim and self.state["selected_nodes"][action] == 0: #if select a node in the left side
                reward = -1
                self.left_selected += 1 

            elif action < self.dim and self.state["selected_nodes"][action] == 1: #if deselect a node in the left side
                reward = 1
                self.left_selected -= 1 

            self.reward = reward #/self.dim

            # update the state
            self.update_state(action)

            # distance is the number of RHS nodes that are not selected
            self.info["curr_energy"] = self.curr_energy
            self.info["num_right_selected"] = self.right_selected
            self.info["num_left_selected"] = self.left_selected

        return [self.state, self.reward, self.done, self.info]


    def update_state(self, action):

        #Ture if node is selected, False if deselected
        selected = (self.state["selected_nodes"][action] == 0) 

        # update the current energy
        self.curr_energy += 1 if selected else -1
        self.state["energy_dist"][1] = self.curr_energy - self.k

        # make the "energy unavailable nodes (deselected nodes due to low energy)" available if the energy is above threshold
        #if self.curr_energy >= self.k and self.curr_energy -1 < self.k and type(self.selected_indexes_prev) != type(None):
        #    self.state["action_mask"][self.selected_indexes_prev] = 1 

        # update the state
        self.state["selected_nodes"][action] = 0 if self.state["selected_nodes"][action] == 1 else 1
        
        if np.array_equal(self.goal, self.state["selected_nodes"]):
            self.done = True
            self.reward += self.dim
        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)

        # update the action mask:

        # find the nieghbours of the new selected\deselected node 
        if action < self.dim: # if action is selecting a LHS node
            ngbrs_lst = self.state["adj_matrix"][action]
            index_constant = self.dim
        else:
            ngbrs_lst = self.state["adj_matrix"][:, self.dim - action]
            index_constant = 0
        ngbrs_index = np.where(ngbrs_lst == 1)[0] + index_constant
        

        # if the node is selected the neighbours become unavailable and vice-versa
        if selected:
            self.state["action_mask"][ngbrs_index] = 0
        else:
            self.state["action_mask"][ngbrs_index] = 1

        # if energy < k, get a negative reward
        if self.curr_energy < self.k:
            self.reward -= self.dim
            #print('hit low energy')

    """
    Loads a random graph from the dataset in form of an adj matrix and sets the threshold k 
    """
    def get_graph(self):
        db = shelve.open(self.dataset)
        dataset_size = db['dataset_size'] 
        graph_size = db['max_graph_size']
        x = np.random.randint(0, dataset_size)
        dic = db[str(x)]

        while True:
            G = dic['graph']
            u, v = dic['size']
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
                #we will just set k to be 2 for now
                self.k =  dic['k'] #u//2 
                break
        db.close()
        return biadj_matrix

    def get_graph_test(self):
        biadj_matrix = np.array([[1., 0., 0., 0., 0.],
                                [1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [1., 1., 1., 1., 0.],
                                [1., 1., 1., 1., 1.]])
        self.k =  4
        return biadj_matrix

    """
    Get the number of nodes (max graph size * 2) in the dataset and set it to n
    """
    def set_n(self):
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
        return [seed]

