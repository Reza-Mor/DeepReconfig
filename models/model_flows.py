
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
import numpy as np
from torch_geometric.nn import NNConv, GCNConv, GATConv, CGConv, GCN
from torch_geometric.nn.pool import global_mean_pool 
from torch_geometric.utils import dense_to_sparse, from_networkx
import torch
import torch.nn as nn
from gymnasium.spaces import Dict
from torch_geometric.data import Data, Batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import MLP
"""
node_features: N x node_dim matrix (N being the number of nodes in the graph)
flows: FG x max_flow_length matrix (FG being the number of flows in the super graph)
Return a 3D |graphs in the batch| x |F| x |node_dim)| matrix (F = FG/|graphs in the batch|)
Each flow embedding is the sum of the node embeddings
"""
def get_flow_embeddings(node_features, flow_paths, num_graphs):
    node_features = F.pad(node_features, (0, 0, 1, 0), mode = 'constant', value = 0)
    # get a |(FG x node_dim)| matrix 
    flow_embeddings = torch.sum(node_features[flow_paths.long()], axis=1)
    # convert to a 2D  |graphs in the batch| x |(F x node_dim)| matrix
    num_flows_per_graph = flow_paths.shape[0]//num_graphs
    flow_embeddings = torch.reshape(flow_embeddings, (num_graphs, num_flows_per_graph, node_features.shape[1]))
    return flow_embeddings

def convert_to_pyG_batch(input_state):
    #print("input_state['node_features'].shape: ", input_state['node_features'].shape)
    num_graphs = input_state['node_features'].shape[0]
    #print("input_state['init_config'].shape: ", input_state['init_config'].shape)
    data_list = []

    #NOTE: check that flows which do not appear in the graph are represented as a vector of shape [1,longest_flow_length] containing only zeros
    num_prev_nodes = 0
    for i in range(num_graphs):
        node_indices_init = (input_state['init_config'][i] != -1).int()
        node_indices_target = (input_state['target_config'][i] != -1).int()
        data_list.append(
            Data(x=input_state['node_features'][i],
                 edge_index= input_state['edge_indices'].type(torch.int64)[i], 
                 edge_features= input_state['edge_features'][i],
                 selected_flows = torch.unsqueeze(input_state['selected_flows'][i], 0) ,
                 init_config = input_state['init_config'][i] + num_prev_nodes * node_indices_init ,
                 target_config = input_state['target_config'][i] + num_prev_nodes * node_indices_target,
                 flow_sizes = input_state['flow_sizes'][i]
                 )
            )
        num_prev_nodes += input_state['node_features'][i].shape[0]
    batch = Batch.from_data_list(data_list = data_list)
    return batch


class GCN_Flows(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, cfg, features_dim=1):
        super().__init__(observation_space, features_dim=1)
        torch.manual_seed(cfg.seed)
        
        self.node_feature_dim = 3
        self.edge_features = 3
        self.out_channels = cfg.out_channels
        self.num_flows = cfg.num_flows #same as number of actions
        #self.num_nodes = 30
        self.graph_dim = cfg.graph_dim
        self._features_dim = self.out_channels + self.num_flows

        # Node level linear layer
        self.nn1 = MLP(in_channels=self.node_feature_dim , out_channels=self.graph_dim, num_layers= 1, norm="batchnorm")
        self.relu = ReLU()

        # GNN Encoder
        self.conv1 = GATConv(in_channels = self.graph_dim, hidden_channels= self.graph_dim, out_channels = self.graph_dim, edge_dim = 3, num_layers = 2, norm="batchnorm")
        
        # GNN Graph Level Decoder
        self.nn2 = Sequential(
            Linear(self.graph_dim, self.out_channels),
            ReLU(),
            #Linear(self.out_channels, self.out_channels),
        )
        

    def forward(self, observations):
        # Data to be passed to the GCN. 
        # Node features are initialized randomly
        # Edge features are the total edge capacity, current free edge capacity , current used edge capacity 
        
        data = convert_to_pyG_batch(observations)
        
        # Pass the node features to simple linear layer
        x = self.nn1(data.x)
        x = self.relu(x)

        # pass the features to the GNN
        x = self.conv1(x, data.edge_index, data.edge_features)

        # get embeddings for the flows
        init_flow_emebeddings = get_flow_embeddings(x, data.init_config, data.num_graphs)
        target_flow_embeddings = get_flow_embeddings(x, data.target_config, data.num_graphs)

        graph_embedding = global_mean_pool(x, data.batch)
        graph_embedding = graph_embedding.unsqueeze(1)

        # get a BS x (2*|number of flows| + 1) x embedding_dim matrix
        x = torch.cat([graph_embedding, init_flow_emebeddings, target_flow_embeddings], axis = 1)
        # pass to the decoder
        x = self.nn2(x)
        
        # obtain a final embedding by taking the mean of graph and flow embedding - TO DO: replace with attention
        x = x.mean(1)
        # include information about which flows are selected
        x = torch.cat([x, data.selected_flows], axis = 1)

        return x

        """
        self._value = self._critic_head(x)
        logits = self._actor_head(x)
        #if data.num_graphs == 1:
            #self._value = self._value.reshape(-1)
            #logits = logits.reshape(-1)
        #print('value: ', self._value.shape)
        #print('logits: ', logits.shape)
        return logits, state
        """

"""
class GCN_Flows(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, lr_schedule, *args, **kwargs):
        
        TorchModelV2.__init__(
            self, obs_space, action_space, lr_schedule, model_config, name, **kwargs
        )
        nn.Module.__init__(self)
        self.var_list = [] # for rllib use

        torch.manual_seed(123)

        node_features = 3
        edge_features = 3
        out_channels = 64 #32
        num_flows = 9  #same as number of actions
        num_nodes = 30
        gnn_out_chanels = 12 

        self.nn1 = Sequential(
            Linear(gnn_out_chanels, 12),
            ReLU(),
            Linear(12, 12),
        )

        # https://github.com/pyg-team/pytorch_geometric/issues/965
        #self.conv1= CGConv(node_features, dim=edge_features, aggr='mean', node_dim=1)
        self.conv1 = GCN(in_channels = node_features, hidden_channels= gnn_out_chanels, out_channels = gnn_out_chanels, num_layers = 2)
        # takes in the graph embeddings, flow_sizes and selected flows, as well as all flow embeddings in the 
        # init and target states

        self._actor_head = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, num_flows)
        )

        self._critic_head = nn.Sequential(
            nn.Linear(12, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        
        # Data to be passed to the GCN. 
        # Node features are initialized randomly
        # Edge features are the total edge capacity, current free edge capacity , current used edge capacity 
        
        #print('rllib data: ', input_dict["obs"])
        data = convert_to_pyG_batch(input_dict["obs"])

        print('pyG data: ', data)
        #print(input_dict)
        #print(input_state)

        # refer to https://github.com/pyg-team/pytorch_geometric/issues/965
        x = self.conv1(data.x, data.edge_index) #data.edge_features)

        init_flow_emebeddings = get_flow_embeddings(x, data.init_config, data.num_graphs)
        target_flow_embeddings = get_flow_embeddings(x, data.target_config, data.num_graphs)

        graph_embedding = global_mean_pool(x, data.batch)
        graph_embedding = graph_embedding.unsqueeze(1)

        # get a BS x (2*|number of flows| + 1) x embedding_dim matrix
        x = torch.cat([graph_embedding, init_flow_emebeddings, target_flow_embeddings], axis = 1)
        #print('x: ', x.shape)
        #x = torch.cat((torch.unsqueeze(action_mask, 2),  torch.unsqueeze(selected_nodes, 2)), 2)
        x = self.nn1(x)
        #print('x1: ', x.shape)
        # obtain a final embedding by taking the mean of graph and flow embedding - TO DO: replace with attention
        x = x.mean(1)
        self._value = self._critic_head(x)
        logits = self._actor_head(x)
        #if data.num_graphs == 1:
            #self._value = self._value.reshape(-1)
            #logits = logits.reshape(-1)
        #print('value: ', self._value.shape)
        #print('logits: ', logits.shape)
        return logits, state

        #self.values = F.softmax(x, dim=1)
        #return self.values, state
    
    def value_function(self):
        #if self.batch_size == 1:
        #    return torch.squeeze(self.values)
        # https://griddly.readthedocs.io/en/latest/rllib/intro/index.html
        return self._value
    
    #def get_action(self, state):
    #    probs = self.forward(state)
    #    highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
    #    log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
    #    return highest_prob_action, log_prob

    def set_adj_matrix_sparse(self, adj_matrix):
        self.adj_matrix = dense_to_sparse(from_networkx(adj_matrix))
"""  

