
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
import numpy as np
from torch_geometric.nn import NNConv, GCNConv, GATConv, CGConv, GCN
from torch_geometric.nn.pool import global_mean_pool 
from torch_geometric.utils import dense_to_sparse, from_networkx
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
torch, nn = try_import_torch()
from gymnasium.spaces import Dict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from torch_geometric.data import Data, Batch


"""
Processes the input data to obtain a list of flow paths
Flow_list: Batch x 
Example: converts [1, 2, 7, 1, 4, 5, 3] with indices [3] to [[1,2,7], [1, 4, 5, 3]].
[[1, 2, 7], [1, 4, 5, 3]] means flow 0 goes through nodes 1,2,3 and flow 1 goes through nodes 1, 4, 5, 3
TO DO: this is slow, replace with something faster
"""
def get_flow_paths_as_list(flow_lst, indices):
    print('flow_list: ', flow_lst)
    print('flow_list: ', flow_lst.shape)
    print('indices: ', indices)
    print('indices: ', indices.shape)
    lst = []
    # iterate over batch, each instance in the batch is a graph with different reconfigs
    for i in range(flow_lst.shape[0]):
        flow_paths = np.split(np.array(flow_lst[i]), indices[i])
        lst.append(list(map(list, flow_paths)))
    # change data type
    return lst
    
"""
node_features: N x node_dim matrix (N being the number of nodes in the graph)
flows: FG x max_flow_length matrix (FG being the number of flows in the super graph)
Return a 3D |graphs in the batch| x |F| x |node_dim)| matrix (F = FG/|graphs in the batch|)
Each flow embedding is the sum of the node embeddings
"""
def get_flow_embeddings(node_features, flow_paths, num_graphs):
    #flow_embeddings = torch.empty(num_graphs, flow_paths.shape[0] * node_features.shape[1])
    #F.pad(node_features, (0, 0, 1, 0, 0, 0), mode = 'constant', value = 0)
        #for i in range(num_graphs):
    #    flow_embeddings[i] = torch.flatten(
    #        torch.sum(node_features[i][flow_paths[i]], axis=1)
    #    )

    node_features = F.pad(node_features, (0, 0, 1, 0), mode = 'constant', value = 0)
    # get a |(FG x node_dim)| matrix 
    flow_embeddings = torch.sum(node_features[flow_paths.long()], axis=1)
    # convert to a 2D  |graphs in the batch| x |(F x node_dim)| matrix
    num_flows_per_graph = flow_paths.shape[0]//num_graphs
    flow_embeddings = torch.reshape(flow_embeddings, (num_graphs, num_flows_per_graph, node_features.shape[1]))
    return flow_embeddings


"""
node_features: Batcn_size x N x node_dim. torch tensor (N being the number of nodes)
flows: F x max_flow_length matrix. torch tensor (F being the number of flows)
Return a 3D Batcn_size x F x node_dim matrix
Each flow embedding is the sum of the node embeddings
"""
def get_flow_embeddings_inv(node_features, flow_paths):
    
    flow_embeddings = torch.sum(node_features[flow_paths], axis=1)
    return torch.flatten(flow_embeddings)

"""
node_features: Batch x N x node_dim matrix (N being the number of nodes)
flow_lst: Batch x 
Return a 2D Batch x (F x node_dim) vector (F being the number of flows)
Each flow embedding is the sum of the node embeddings
"""
def get_flow_embeddings2(node_features, flow_lst, indices):
    batch_flow_embeddings = torch.empty(node_features.shape[0], indices[1] * node_features.shape[2])
    # iterate over batch, each instance in the batch is a graph with different reconfigs
    for i in range(flow_lst.shape[0]):
        flow_paths = np.split(np.array(flow_lst[i]), indices[i])
        flow_paths = list(map(list, flow_paths))
        graph_flow_embeddings = torch.sum(node_features[flow_paths], axis=1)

def convert_to_pyG_batch(input_state):
    num_graphs = input_state['node_features'].shape[0]
    print("input_state['init_config'].shape: ", input_state['init_config'].shape)
    data_list = []

    #NOTE: check that flows which do not appear in the graph are represented as a vector of shape [1,longest_flow_length] containing only zeros
    num_prev_nodes = 0
    for i in range(num_graphs):
        non_zero_indices_init = (input_state['init_config'][i] != 0).int()
        non_zero_indices_target = (input_state['target_config'][i] != 0).int()
        data_list.append(
            Data(x=input_state['node_features'][i],
                 edge_index= input_state['edge_indices'].type(torch.int64)[i], 
                 edge_features= input_state['edge_features'][i],
                 selected_flows = input_state['selected_flows'][i],
                 init_config = input_state['init_config'][i] + num_prev_nodes * non_zero_indices_init ,
                 target_config = input_state['target_config'][i] + num_prev_nodes * non_zero_indices_target,
                 flow_sizes = input_state['flow_sizes'][i]
                 )
            )
        num_prev_nodes += input_state['node_features'][i].shape[0]
    batch = Batch.from_data_list(data_list = data_list)
    return batch


#class GCN(torch.nn.Module):
class GCN_Flows(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
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
        

        data = convert_to_pyG_batch(input_dict["obs"])

        print(data)
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
        print('x: ', x.shape)
        #x = torch.cat((torch.unsqueeze(action_mask, 2),  torch.unsqueeze(selected_nodes, 2)), 2)
        x = self.nn1(x)
        print('x1: ', x.shape)
        # obtain a final embedding by taking the mean of graph and flow embedding - TO DO: replace with attention
        x = x.mean(1)
        self._value = self._critic_head(x)
        logits = self._actor_head(x)
        #if data.num_graphs == 1:
            #self._value = self._value.reshape(-1)
            #logits = logits.reshape(-1)
        print('value: ', self._value.shape)
        print('logits: ', logits.shape)
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
        

