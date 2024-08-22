
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MLP, GCN
from torch_geometric.utils import dense_to_sparse, from_networkx
from gymnasium.spaces import Dict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from torch_geometric.data import Data, Batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import MLP

def convert_to_pyG_batch(input_state, cfg):
    #print("input_state['node_features'].shape: ", input_state['node_features'].shape)
    num_graphs = input_state['selected_left_nodes'].shape[0]
    num_nodes = input_state['selected_left_nodes'].shape[1] # number of nodes on one side of the graph
    #print("input_state['init_config'].shape: ", input_state['init_config'].shape)
    data_list = []

    for i in range(num_graphs):
        data_list.append(
            Data(#x=input_state['node_features'][i],
                 edge_index= input_state['edge_indices'].type(torch.int64)[i], 
                 selected_nodes = torch.cat((input_state['selected_left_nodes'][i], input_state['selected_right_nodes'][i]), dim=0).to(torch.int32),
                 action_mask = torch.cat((torch.zeros(num_nodes).to(cfg.device), input_state['action_mask'][i]), dim=0).to(torch.int32),
                 #selected_nodes = torch.unsqueeze(torch.cat((input_state['selected_left_nodes'][i], input_state['selected_right_nodes'][i]), dim=0), dim =0),
                 #action_mask = torch.unsqueeze(torch.cat((torch.zeros(num_nodes).to(cfg.device), input_state['action_mask'][i]), dim=0),
                 energy_dist = torch.unsqueeze(input_state['energy_dist'][i], 0)
                 )
            )
    batch = Batch.from_data_list(data_list = data_list)
    return batch


class GCN_VC(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, cfg, features_dim=1):
        super().__init__(observation_space, features_dim=1)
        torch.manual_seed(cfg.seed)
        
        self.cfg = cfg
        self.out_channels = cfg.out_channels
        self.num_nodes = cfg.num_nodes #same as number of actions
        #self.num_nodes = 30
        self.graph_dim = cfg.graph_dim

        self._features_dim = self.out_channels + 2

        # Node level linear layer, the first two bits represent if the node is selected, the second two bits represent the action mask
        self.nn1 = nn.Embedding(2, int(self.graph_dim//2))
        self.nn2 = nn.Embedding(2, int(self.graph_dim//2))
        self.relu = ReLU()

        # GNN Encoder
        self.conv1 = GCN(in_channels = self.graph_dim, hidden_channels= self.graph_dim, out_channels = self.graph_dim, num_layers = 2, norm="batchnorm")
        
        # GNN Graph Level Decoder
        self.nn3 = Sequential(
            Linear(self.graph_dim, self.out_channels),
            ReLU(),
            #Linear(self.out_channels, self.out_channels),
        )
        

    def forward(self, observations):
        data = convert_to_pyG_batch(observations, self.cfg)
        
        # get node level emebeddings
        #print('data.selected_left_nodes: ', data.selected_left_nodes.shape)
        #selected_nodes = torch.cat((data.selected_left_nodes, data.selected_right_nodes), dim=0).to(torch.int32)
        #print('selected_nodes: ', selected_nodes.shape)
        #right_nodes_action_mask = torch.zeros(data.num_graphs * self.num_nodes).to(self.cfg.device)
        #action_mask = torch.cat((right_nodes_action_mask, data.action_mask), dim=0).to(torch.int32)
        #print('action_mask: ', action_mask.shape)
        
        x = torch.cat((self.nn1(data.selected_nodes), self.nn2(data.action_mask)), dim=1)
        x = self.relu(x)

        # pass the features to the GNN
        x = self.conv1(x, data.edge_index)

        graph_embedding = global_mean_pool(x, data.batch)
 
        # pass to graph decoder
        x = self.nn3(graph_embedding)
        x = self.relu(x)

        # include information about the energy distance
        energy_norm = data.energy_dist / self.num_nodes
        x = torch.cat((x, energy_norm), 1)
 
        return x
