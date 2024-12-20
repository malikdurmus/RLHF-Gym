import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.envs.mujoco import MujocoEnv

# Params Actor
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class EstimatedRewardNetwork(nn.Module):
    """
    Neural network for reward estimation
    """
    #Neural network for reward estimation
    def __init__(self, env: MujocoEnv):
        """
        :param env: Instance of MujocoEnv (strict), which provides the observation_space and action_space properties. These describe the dimensions of the environment's observations and actions. This parameter is used to define the input size for the first fully connected layer.
        """
        super().__init__()
        self.fc1 = nn.Linear( np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape),
                    256) #observation_space and action space shape differ greatly among envs. we could define a function to overcome overfitting/underfitting
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)


    def forward(self, action, observation):
        """
        :param action: Tensor representing the actions taken, expected to be concatenated with the observation.
        :param observation: Tensor representing the state observations from the environment.
        :return: Tensor representing the estimated reward as predicted by the network.
        """
        # Concatenate states and actions
        x = torch.cat([action, observation], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward




# Q-Network
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



class Actor(nn.Module):
    # Neural network architecture
    def __init__(self, env):
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
        x = F.relu(self.fc1(x)) # first layer - ReLU
        x = F.relu(self.fc2(x)) # second layer - ReLU
        mean = self.fc_mean(x) # calculate mean
        log_std = self.fc_logstd(x)                                                 # log-standard deviation
        log_std = torch.tanh(log_std)                                               # restrict to [-1, 1]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)   # scale to [MIN, MAX]
        return mean, log_std

    # state -> action, log-probability, mean
    def get_action(self, x):
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


def initialize_networks(envs, device, policy_lr, q_lr,reward_model_lr):
    actor = Actor(envs).to(device)
    # Reward network
    reward_network = EstimatedRewardNetwork(envs).to(device)
    # Q-Networks
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    # Sync target networks
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # Optimizer q-function (Critic)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
    # Optimizer Actor (policy)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)

    return actor, reward_network ,qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer