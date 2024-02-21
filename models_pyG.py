
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse, from_networkx
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
torch, nn = try_import_torch()
from gymnasium.spaces import Dict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC


#class GCN(torch.nn.Module):
class GCN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)
        self.var_list = []

        torch.manual_seed(123)

        #model = Sequential('x, edge_index, batch', [
        #        (GCNConv(num_features, 64), 'x, edge_index -> x1'),
        #        ReLU(inplace=True),
        #        (GCNConv(64, 64), 'x1, edge_index -> x2'),
        #        ReLU(inplace=True),
        #        Linear(2 * 64, dataset.num_classes),
        #    ])
        
        #https://github.com/pyg-team/pytorch_geometric/issues/965

        #TO DO: checnge 5 to the input number of features
        self.conv1 = GCNConv(5, 32, node_dim=1)
        self.conv2 = GCNConv(32, 64, node_dim=1)
        self.linear1 = Linear(64 + 2, 32)
        #self.linear2 = Linear(32, 20)
        self.relu = ReLU()
        #self.linear2 = Linear(32, 20)

        #sparse adj matrix, same as edge index
        self.adj_matrix_sparse = None 

        
        self._actor_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 20)
        )

        self._critic_head = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        
        input_state = input_dict["obs"]
        
        #print('input_dict: ', input_dict)
        #print('input_dict: ', input_dict.get("obs"))
        #print('input_state: ', input_state)

        #restore_original_dimensions(input_dict["obs"], self.obs_space, "torch")
        #print('node_features: ', input_state['node_features'].shape)
        #print('edge_indices: ', input_state['edge_indices'].shape)
        #print('selected_left_nodes: ', input_state['selected_left_nodes'].shape)
        #print('action_mask: ', input_state['action_mask'].shape)

        # NOTE: the batch size refers to the number of graphs (configurations), not the number of nodes
        # refer to https://github.com/pyg-team/pytorch_geometric/issues/965

        self.batch_size = input_state['action_mask'].shape[0]
        
        #extract the features
        n = input_state['selected_left_nodes'].size()[1]
        selected_nodes = torch.cat((input_state['selected_left_nodes'], input_state['selected_right_nodes']), 1)
        #print(torch.zeros(n).shape)
        action_mask = torch.cat((torch.zeros((self.batch_size, n)), input_state['action_mask']), 1)
        node_features = input_state['node_features']
        
        #print('node_features: ', node_features.shape)
        #print('action_mask: ', action_mask.shape)
        #print('selected_nodes: ', selected_nodes.shape)

        x = torch.cat((torch.unsqueeze(action_mask, 2),  torch.unsqueeze(selected_nodes, 2)), 2)
        #print('x: ', x.shape)
        x = torch.cat((node_features, x), 2)
        #print('x: ', x.shape)

        edge_indices = input_state['edge_indices'].type(torch.int64) 
        #print('edge_indices: ', edge_indices.shape)
        #print('edge_indices: ', edge_indices[0].shape)

        x = self.conv1(x, edge_indices[0])
        x = self.relu(x)
        x = self.conv2(x, edge_indices[0])
        #mean pooling
        #print('x: ', x.shape)
        x = x.mean(1)
        #print('x: ', x.shape)
        x = torch.cat((x, input_state['energy_dist']), 1)
        x = self.linear1(x)
        #print('x: ', x.shape)
        x = self.relu(x)
        #x = self.linear2(x)

        value = self._critic_head(x)
        self._value = value.reshape(-1)
        logits = self._actor_head(x)
        return logits, state

        #self.values = F.softmax(x, dim=1)
        #return self.values, state
    
    def value_function(self):
        #print(self.values)
        #if self.batch_size == 1:
        #    return torch.squeeze(self.values)
        # https://griddly.readthedocs.io/en/latest/rllib/intro/index.html
        return self._value
    
    def get_action(self, state):
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def set_adj_matrix_sparse(self, adj_matrix):
        self.adj_matrix = dense_to_sparse(from_networkx(adj_matrix))
        


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        #orig_space = getattr(obs_space, "original_space", obs_space)
        #assert (
        #    isinstance(orig_space, Dict)
        #    and "action_mask" in orig_space.spaces
        #    and "observations" in orig_space.spaces
        #)
        self.var_list = []
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()