import torch.optim as optim
from .actor import Actor
from .critic import SoftQNetwork
from .reward_networks import EstimatedRewardNetwork

def initialize_networks(envs, device, policy_lr, q_lr, num_models):
    actor = Actor(envs).to(device)
    reward_networks = [EstimatedRewardNetwork(envs).to(device) for _ in range(num_models)]
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)
    return actor, reward_networks, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer