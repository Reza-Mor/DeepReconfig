from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
import numpy as np
from ray.rllib.agents.dqn import dqn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from spektral.layers import GATConv
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.regularizers import l2
from ray.rllib.policy.sample_batch import SampleBatch
#from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from ray.rllib.utils.torch_utils import FLOAT_MIN
from scipy.sparse import coo_matrix
#torch, nn = try_import_torch()
#tf, tf_original, tf_version = try_import_tf(error = True)
import tensorflow as tf
#from ray.rllib.models.tf.random import set_seed

MODEL_CONFIG_1 = {"custom_model": "FeedForward", 
                       "custom_model_config": {"fcnet_hiddens": [64, 32, 32], "fcnet_activation": "relu", "no_final_linear": False}}
                       
MODEL_CONFIG_2 = {"custom_model": "FeedForward", 
                       "custom_model_config": {"fcnet_hiddens": [128, 128, 64, 64], "fcnet_activation": "relu", "no_final_linear": False}}

MODEL_CONFIG_3 = {"custom_model": "GAT", 
                       "custom_model_config": {"fcnet_hiddens": [256, 128, 64], "fcnet_activation": "relu", "no_final_linear": False}}
#set_seed(0)

class FeedForward(TFModelV2):
    """
    Feed Forward Model that handles simple discrete action masking.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        #action_mask = input_dict["obs"]["action_mask"]

        orig_obs_flat = input_dict["obs_flat"]
        orig_obs = input_dict["obs"]
        print(orig_obs)

        #action_mask = orig_obs["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": orig_obs_flat})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        #inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        #masked_logits = logits + inf_mask

        # Return masked logits.
        #return masked_logits, state
        return logits, state

    def value_function(self):
        return self.internal_model.value_function()


class GNN(TFModelV2):
    """
    GNN Model that handles simple discrete action masking.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        # number of nodes in the graph
        self.N = 40
        # number of features per node
        self.F = 64
        # initial features
        self.h_init = np.random.uniform(0,1,(self.N,self.F))

        self.x_in = Input(shape=(self.F,))
        self.a_in = Input((self.N,), sparse=True)

        n_out = 8  # Number of n_out in each head of the first GAT layer
        n_attn_heads = 8  # Number of attention heads in first GAT layer
        dropout = 0.9  # Dropout rate for the features and adjacency matrix
        l2_reg = 2.5e-4  # L2 regularization rate

        self.gc_1 = GATConv(
            n_out,
            attn_heads=n_attn_heads,
            concat_heads=True,
            dropout_rate=dropout,
            activation="elu",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            )

        self.internal_model = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)


    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        #action_mask = input_dict["obs"]["action_mask"]

        orig_obs_flat = input_dict["obs_flat"]
        orig_obs = input_dict["obs"]
        action_mask = orig_obs["action_mask"]

        self.gc_1([self.x_in, self.a_in])

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": orig_obs_flat})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class GAT(TFModelV2):
    """
    GNN Model that handles simple discrete action masking.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        # number of nodes in the graph
        self.N = 40
        # number of features per node
        self.F = 64
        # initial features
        np.random.seed(0)
        self.x = tf.constant(np.random.uniform(0,1,(self.N, self.F)))

        self.x_in = Input(shape=(self.F,))
        self.a_in = Input((self.N,), sparse=True)

        self.channels_1 = 64 # Number of channels (output) in each head of the first GAT layer
        #self.channels_2 = 64 #obs_space.shape  # Number of channels (output) in each head of the first GAT layer
        n_attn_heads = 8  # Number of attention heads in first GAT layer
        dropout = 0.9  # Dropout rate for the features and adjacency matrix
        l2_reg = 2.5e-4  # L2 regularization rate

        self.gc_1 = GATConv(
            self.channels_1,
            attn_heads=n_attn_heads,
            concat_heads=True,
            dropout_rate=dropout,
            activation="elu",
            kernel_regularizer=l2(l2_reg),
            attn_kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            add_self_loops = True,
            )

        #self.dense_1 = tf.keras.layers.Dense(channels_2 * self.N, activation='relu')

        #self.dense_2 = tf.keras.layers.Dense(self.N, activation='relu')

        #self.softmax
        
        self.internal_model = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        #action_mask = input_dict["obs"]["action_mask"]

        orig_obs_flat = input_dict["obs_flat"]
        orig_obs = input_dict["obs"]
        action_mask = orig_obs["action_mask"]
        adj_matrix = orig_obs["adj_matrix"]
        selected_right_nodes = orig_obs["selected_right_nodes"]
        deselected_right_nodes = abs(1 - orig_obs["selected_right_nodes"])
        selected_left_nodes = orig_obs["selected_left_nodes"]
        deselected_left_nodes = abs(1 - orig_obs["selected_left_nodes"])
        energy_dist = orig_obs["energy_dist"]

        # pass the node embeddings and adjacency matrix to the graph encoder

        # node features of shape `([batch], n_nodes, n_node_features)`- batch size is 1
        # binary adjacency matrix of shape `([batch], n_nodes, n_nodes)` - batch size is 1

        row, colmn = tf.experimental.numpy.nonzero(adj_matrix)[0], tf.experimental.numpy.nonzero(adj_matrix)[1]
        if row.shape[0] != 0:
            # unweighted matrix
            data = np.array(row.shape[0]*[1])
            sparse_matrix = coo_matrix((data, (row, col))) # turn this to scipy sparse matrix
        else:
            #zero = tf.constant([0])
            zero = np.array([0])
            sparse_matrix = coo_matrix((zero, (zero, zero)))
        sparse_matrix = sp_matrix_to_sp_tensor(sparse_matrix)
        #a = self.a_in(sparse_matrix, sparse=True)

        #x = self.x_in(self.x)

        # out is a matrix with dim ([batch], n_nodes, channels)
        out = self.gc_1([self.x, sparse_matrix])
        out =  self.gc_1([out, sparse_matrix])

        # get rid of the batch (1) dimension
        out = tf.squeeze(out)

        # get the context vectors for the decoder
        selected_right_indexes = np.nonzero(selected_right_nodes)
        deselected_right_indexes = np.nonzero(deselected_right_nodes)
        selected_left_indexes = np.nonzero(selected_left_nodes)
        deselected_left_indexes = np.nonzero(deselected_left_nodes)

        selected_r_embeddings = tf.gather(out, selected_right_indexes , axis=0) if selected_right_indexes != [] else tf.constant(self.channels_1 * [0])
        deselected_r_embeddings = tf.gather(out, deselected_right_indexes, axis=0) if deselected_right_indexes != [] else tf.constant(self.channels_1 * [0])
        selected_l_embeddings = tf.gather(out, selected_left_indexes , axis=0) if selected_left_indexes != [] else tf.constant(self.channels_1 * [0])
        deselected_l_embeddings = tf.gather(out, deselected_left_indexes, axis=0) if deselected_left_indexes != [] else tf.constant(self.channels_1 * [0])

        if len(selected_right_indexes) == 1:
            selected_r_embeddings = selected_r_embeddings[0]
        if len(deselected_right_indexes) == 1:
            deselected_r_embeddings = deselected_r_embeddings[0]
        if len(selected_left_indexes) == 1:
            selected_l_embeddings = selected_l_embeddings[0]
        if len(deselected_left_indexes) == 1:
            deselected_l_embeddings = deselected_l_embeddings[0]

        energy_dist = tf.convert_to_tensor(energy_dist, dtype=tf.float32)
        h_S = tf.reduce_sum(selected_r_embeddings, axis = 0)
        h_S_c = tf.reduce_sum(deselected_r_embeddings, axis = 0)
        h_L = tf.reduce_sum(selected_l_embeddings, axis = 0)
        h_L_c = tf.reduce_sum(deselected_l_embeddings, axis = 0)
        h_G = tf.reduce_sum(out, axis = 0)
        context = tf.concat([h_G, h_S, h_L, h_L_c, h_S_c, energy_dist], axis = -1)

        # Compute the unmasked logits.
        # logits, _ = self.internal_model({"obs": orig_obs_flat})
        logits, _ = self.internal_model({"obs": context})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()