import gym
from gym.utils import seeding
from gym.spaces import Space, Box, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import math
from networkx.algorithms import bipartite
import shelve

class Rnaenv_v5 (gym.Env):

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
            "action_mask": MultiBinary(self.n),
            "selected_right_nodes_1": MultiBinary(self.n//2),
            "selected_left_nodes_1": MultiBinary(self.n//2),
            "energy_dist_1": Box(-1, self.n, shape=(2,), dtype=np.int32),
            "selected_right_nodes_2": MultiBinary(self.n//2),
            "selected_left_nodes_2": MultiBinary(self.n//2),
            "energy_dist_2": Box(-1, self.n, shape=(2,), dtype=np.int32)
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
        adj_matrix = self.get_graph()
        action_mask = np.ones(self.n, dtype=np.int32)
        selected_left_nodes_1 = np.ones(self.dim, dtype=np.int32)
        selected_right_nodes_1 = np.zeros(self.dim, dtype=np.int32)
        # at first, we can only deselect the nodes on the LHS
        self.curr_energy_1 = self.dim #the current energy is simply the number of selected nodes

        selected_left_nodes_2 = np.zeros(self.dim, dtype=np.int32)
        selected_right_nodes_2 = np.ones(self.dim, dtype=np.int32)
        # at first, we can only deselect the nodes on the LHS
        self.curr_energy_2 = self.dim #the current energy is simply the number of selected nodes

        self.init_state = {"adj_matrix": adj_matrix,
                        "action_mask": action_mask,
                        "selected_right_nodes_1": selected_right_nodes_1,
                        "selected_left_nodes_1": selected_left_nodes_1,
                        "energy_dist_1": np.array([self.k_1, self.curr_energy_1 - self.k_1], dtype= np.int32),
                        "selected_right_nodes_2": selected_right_nodes_2,
                        "selected_left_nodes_2": selected_left_nodes_2,
                        "energy_dist_2": np.array([self.k_2, self.curr_energy_2 - self.k_2], dtype= np.int32)}

        #self.goal = np.ones(self.dim, dtype=np.int32)       
        self.count = 0
        self.state = self.init_state
        self.reward = 0
        #self.max_selected_right = 0
        self.selected_indexes_prev_1 = None #this is used for energy level implementation
        self.selected_indexes_prev_2 = None
        self.selected_1 = None #True if the current action is selecting a node
        self.selected_2 = None #True if the current action is selecting a node
        self.done = False
        self.info = {}
        self.distance_prev = self.dim
        self.distance = self.dim

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

            if action < self.dim: 
                self.update_state_1(action)
            else: 
                action = action - self.dim
                self.update_state_2(action)
           
            try:
                assert self.observation_space.contains(self.state)
            except AssertionError:
                print("INVALID STATE", self.state)
            
            self.distance_prev = self.distance
            self.distance = abs(self.state["selected_right_nodes_1"] - self.state["selected_right_nodes_2"]).sum()
            self.reward = self.distance_prev - self.distance

            if np.array_equal(self.state["selected_right_nodes_1"], self.state["selected_right_nodes_2"]):
                self.done = True
                self.reward += self.dim

        return [self.state, self.reward, self.done, self.info]


    def update_state_1(self, action):

        #Ture if node is selected, False if deselected
        self.selected_1 = (self.state["selected_right_nodes_1"][action] == 0)

        # find the nieghbours of the new selected\deselected node 
        ngbrs_lst = self.state["adj_matrix"][:, action]
        ngbrs_index = np.where(ngbrs_lst == 1)[0]

        # self.curr_energy - self.k >= num_prev_selected_left_nodes checks if the threshold will not go under k after the selection
        prev_selected_left_nodes = self.state["selected_left_nodes_1"] * ngbrs_lst
        num_prev_selected_left_nodes = sum(prev_selected_left_nodes)

        if self.selected_1 and self.curr_energy_1 - self.k_1 >= num_prev_selected_left_nodes:
            self.curr_energy_1 -= (num_prev_selected_left_nodes - 1)
            self.state["selected_left_nodes_1"][ngbrs_index] = 0
            self.state["selected_right_nodes_1"][action] = 1

        elif not self.selected_1:

            # update the state
            self.state["selected_right_nodes_1"][action] = 0

            if ngbrs_index.shape != (0,):
                # ngbrs_rows: each row is a neighbour n of the RHS node selected
                # each cloumn is the neighbours of the node n |neighbours| by |RHS| matrix
                ngbrs_rows = self.state["adj_matrix"][ngbrs_index, :]
                # m[i,j] is 1 iff the j-th node in RHS is selected and is a neighnbour of node i in LHS 
                m = np.multiply(ngbrs_rows, self.state["selected_right_nodes_1"])
                # mask[i] is true if node i in LHS can be selected (all the neighbours of the i-th node in RHS are not selected)
                mask = abs(1 - np.any(m, axis = 1))
                # indexes of neighbour nodes in LHS which have no selected neighbour in RHS
                indexs = np.multiply(ngbrs_index, mask)
                self.state["selected_left_nodes_1"][indexs] = 1
                self.curr_energy_1 += (mask.sum() - 1)
            
        # make the "energy unavailable nodes (deselected nodes due to low energy)" available if the energy is above threshold
        if self.curr_energy_1 >= self.k_1 and type(self.selected_indexes_prev_1) != type(None):
            self.state["action_mask"][:self.dim][self.selected_indexes_prev_1] = 1

        # if energy < k, cannot deselect any selected nodes
        elif self.curr_energy_1 < self.k_1:
            selected_indexes = np.where(self.state["selected_right_nodes_1"] == 1)[0]
            self.selected_indexes_prev_1 =  selected_indexes.copy()
            self.state["action_mask"][:self.dim][selected_indexes] = 0
            #print('hit low energy')

        # distance is the number of RHS nodes that are not selected
        self.info["curr_energy_1"] = self.curr_energy_1
        self.info["num_right_selected_1"] = self.state["selected_right_nodes_1"].sum()
        self.info["num_left_selected_1"] = self.state["selected_left_nodes_1"].sum()

    def update_state_2(self, action):

        #Ture if node is selected, False if deselected
        self.selected_2 = (self.state["selected_right_nodes_2"][action] == 0)

        # find the nieghbours of the new selected\deselected node 
        ngbrs_lst = self.state["adj_matrix"][:, action]
        ngbrs_index = np.where(ngbrs_lst == 1)[0]

        # self.curr_energy - self.k >= num_prev_selected_left_nodes checks if the threshold will not go under k after the selection
        prev_selected_left_nodes = self.state["selected_left_nodes_2"] * ngbrs_lst
        num_prev_selected_left_nodes = sum(prev_selected_left_nodes)

        if self.selected_2 and self.curr_energy_2 - self.k_2 >= num_prev_selected_left_nodes:
            self.curr_energy_2 -= (num_prev_selected_left_nodes - 1)
            self.state["selected_left_nodes_2"][ngbrs_index] = 0
            self.state["selected_right_nodes_2"][action] = 1

        elif not self.selected_2:

            # update the state
            self.state["selected_right_nodes_2"][action] = 0

            if ngbrs_index.shape != (0,):
                # ngbrs_rows: each row is a neighbour n of the RHS node selected
                # each cloumn is the neighbours of the node n |neighbours| by |RHS| matrix
                ngbrs_rows = self.state["adj_matrix"][ngbrs_index, :]
                # m[i,j] is 1 iff the j-th node in RHS is selected and is a neighnbour of node i in LHS 
                m = np.multiply(ngbrs_rows, self.state["selected_right_nodes_2"])
                # mask[i] is true if node i in LHS can be selected (all the neighbours of the i-th node in RHS are not selected)
                mask = abs(1 - np.any(m, axis = 1))
                # indexes of neighbour nodes in LHS which have no selected neighbour in RHS
                indexs = np.multiply(ngbrs_index, mask)
                self.state["selected_left_nodes_2"][indexs] = 1
                self.curr_energy_2 += (mask.sum() - 1)

        # make the "energy unavailable nodes (deselected nodes due to low energy)" available if the energy is above threshold
        if self.curr_energy_2 >= self.k_2 and type(self.selected_indexes_prev_2) != type(None):
            self.state["action_mask"][self.dim:][self.selected_indexes_prev_2] = 1

        # if energy < k, cannot deselect any selected nodes
        elif self.curr_energy_2 < self.k_2:
            selected_indexes = np.where(self.state["selected_right_nodes_2"] == 1)[0]
            self.selected_indexes_prev_2 =  selected_indexes.copy()
            self.state["action_mask"][self.dim:][selected_indexes] = 0
            #print('hit low energy')

        # distance is the number of RHS nodes that are not selected
        self.info["curr_energy_2"] = self.curr_energy_2
        self.info["num_right_selected_2"] = self.state["selected_right_nodes_2"].sum()
        self.info["num_left_selected_2"] = self.state["selected_left_nodes_2"].sum()

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
                self.k_1 =  dic['k'] #u//2 
                self.k_2 =  dic['k']
                break
        db.close()
        return biadj_matrix


    def get_graph_test(self):
        biadj_matrix = np.array([[1, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0]])
        self.k_1 =  4
        self.k_2 =  4
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

