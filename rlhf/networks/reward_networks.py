import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EstimatedRewardNetwork(nn.Module):
    """
    Neural network for reward estimation
    """
    #Neural network for reward estimation
    def __init__(self, env):
        """
        :param env: Instance of MujocoEnv (strict), which provides the observation_space and action_space properties. These describe the dimensions of the environment's observations and actions. This parameter is used to define the input size for the first fully connected layer.
        """
        super().__init__()
        self.fc1 = nn.Linear( np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape),
                    256) #TODO: observation_space and action space shape differ greatly among envs. we could define a function to overcome overfitting/underfitting after testing on training data but vs on validation or test data
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, action, observation):
        """
        :param action: Tensor representing the actions taken, expected to be concatenated with the observation.
        :param observation: Tensor representing the state observations from the environment.
        :return: Tensor representing the estimated reward as predicted by the network.
        """
        # Concatenate states and actions
        action = action.squeeze(1)  # resulting in shape [15, 6]
        observation = observation.squeeze(1)  # shape [15, 18]
        x = torch.cat([action, observation], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        reward = self.fc3(x)


        return reward

