import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """
    Actor network for the algorithm.
    """
    # Neural network architecture
    def __init__(self, env):
        """
        Initialize the Actor network.
        Two fully connected standard layers.
        One layer to calculate the mean of the action distribution, another one to calculate the log-standard deviation
        of the action distribution.
        :param env: The specific environment from which we derive the observation and action space.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256) # first layer
        self.fc2 = nn.Linear(256, 256) # second layer

        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape)) # mean
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape)) # log-standard deviation

        # Rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    # state -> mean, log-standard deviation
    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input state tensor.
        :return: mean: Mean of the action distribution
                 log_std: Log-standard deviation of the action distribution
        """
        x = F.relu(self.fc1(x)) # first layer - ReLU
        x = F.relu(self.fc2(x)) # second layer - ReLU
        mean = self.fc_mean(x) # calculate mean
        log_std = self.fc_logstd(x)                                                 # log-standard deviation
        log_std = torch.tanh(log_std)                                               # restrict to [-1, 1]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)   # scale to [MIN, MAX]
        return mean, log_std

    # state -> action, log-probability, mean
    def get_action(self, x):
        """
        Samples an action from the action distribution and compute its log-probability.
        :param x: Input state tensor.
        :return: action: the sampled action.
                 log_prob: Log probability of the sampled action
                 mean: mean of the action distribution (the most likely action)
        """
        mean, log_std = self(x) # forward -> mean, log-standard deviation
        std = log_std.exp() # convert log-standard deviation -> standard deviation
        normal = torch.distributions.Normal(mean, std) # Gaussian distribution
        x_t = normal.rsample() # reparametrize Sample (?)
        y_t = torch.tanh(x_t) # restrict to [-1, 1]
        action = y_t * self.action_scale + self.action_bias # scale to valid scope of actions
        log_prob = normal.log_prob(x_t)                                     # log-probabibility
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)  # Jacobi
        log_prob = log_prob.sum(1, keepdim=True)                            # sum of log-probabilities
        mean = torch.tanh(mean) * self.action_scale + self.action_bias # calculate mean of action distribution
        return action, log_prob, mean

