from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
import numpy as np
from ray.rllib.agents.dqn import dqn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
#from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
torch, nn = try_import_torch()
tf, tf_original, tf_version = try_import_tf(error = True)


MODEL_CONFIG_1 = {"custom_model": "CustomModel", 
                       "custom_model_config": {"fcnet_hiddens": [64, 32, 32], "fcnet_activation": "relu", "no_final_linear": False}}
                       

MODEL_CONFIG_2 = {"custom_model": "CustomModel", 
                       "custom_model_config": {"fcnet_hiddens": [64, 32, 32, 32], "fcnet_activation": "relu", "no_final_linear": False}}

class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
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
        action_mask = orig_obs["action_mask"]

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