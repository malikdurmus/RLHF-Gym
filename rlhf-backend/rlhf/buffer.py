import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

# Replay Buffer
def initialize_rb(envs, buffer_size, device):
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    return rb