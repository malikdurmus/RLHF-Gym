import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftQNetwork(nn.Module):
    # Neural network architecture
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256) # first layer
        self.fc2 = nn.Linear(256, 256) # second layer
        self.fc3 = nn.Linear(256, 1) # third layer -> unique value

    # Q(s,a) -> Q-value
    def forward(self, x, a):
        x = torch.cat([x, a], 1) # Concatenate state, action
        x = F.relu(self.fc1(x)) # first layer - ReLU
        x = F.relu(self.fc2(x)) # second layer - ReLU
        x = self.fc3(x) # third layer -> Q-value
        return x

