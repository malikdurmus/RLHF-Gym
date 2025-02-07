import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftQNetwork(nn.Module):
    """
    Soft Q-Network (Critic) for estimating the Q-value of state-action pairs.
    """
    # Neural network architecture
    def __init__(self, env):
        """
        Initializes the Soft Q-Network.
        :param env: The environment from which observation and action spaces are derived. (Mujoco)
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256) # first layer
        self.fc2 = nn.Linear(256, 256) # second layer
        self.fc3 = nn.Linear(256, 1) # third layer -> unique value

    # Q(s,a) -> Q-value
    def forward(self, x, a):
        """
        Forward pass through the network to compute the Q-value for a given state-action pair
        :param x: State Tensor.
        :param a: Action Tensor.
        :return: Q-Value for our state/action pair (Tensor).
        """
        x = torch.cat([x, a], 1) # Concatenate state, action
        x = F.relu(self.fc1(x)) # first layer - ReLU
        x = F.relu(self.fc2(x)) # second layer - ReLU
        x = self.fc3(x) # third layer -> Q-value
        return x

