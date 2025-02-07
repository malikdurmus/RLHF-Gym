import torch.optim as optim
from .actor import Actor
from .critic import SoftQNetwork
from .reward_networks import EstimatedRewardNetwork

def initialize_networks(envs, device, policy_lr, q_network_lr, num_models):
    """
    Initialize all of our networks.
    :param envs: The environment from which observation and action spaces are derived. (Mujoco)
    :param device: "cuda" or "cpu" for computation.
    :param policy_lr: Learning rate of our Actor.
    :param q_network_lr: Learning rate of our Critics.
    :param num_models: number of reward networks.
    :return: All the networks.
    """
    actor = Actor(envs).to(device)
    reward_networks = [EstimatedRewardNetwork(envs).to(device) for _ in range(num_models)]
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_network_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)
    return actor, reward_networks, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer