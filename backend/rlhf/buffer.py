import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from collections import namedtuple

# Define TrajectorySamples at the module level
TrajectorySamples = namedtuple("TrajectorySamples", ["states", "actions", "rewards"])

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

class PreferenceBuffer:
    def __init__(self, buffer_size, device):
        self.buffer = []
        self.buffer_size = buffer_size
        self.device = device

    def add(self, trajectories, preference):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append([trajectories, preference])

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

class TrajectorySampler:
    def __init__(self, rb: ReplayBuffer):
        self.rb = rb

    # Single trajectory
    def uniform_trajectory(self, traj_length, time_window):
        if self.rb.size() < traj_length or self.rb.size() < time_window or time_window < traj_length:
            raise ValueError("Not enough data to sample")

        # Random start index (exclude end of buffer)
        min_start_index = self.rb.size() - time_window + 1
        max_start_index = self.rb.size() - traj_length + 1
        start_index = np.random.randint(min_start_index, max_start_index)
        end_index = start_index + traj_length

        # Extract states, actions, rewards
        states = torch.tensor(self.rb.observations[start_index:end_index])
        actions = torch.tensor(self.rb.actions[start_index:end_index])
        rewards = torch.tensor(self.rb.rewards[start_index:end_index])

        # Name tensors for better access
        trajectory = TrajectorySamples(
            states=states if states.ndim > 1 else states.unsqueeze(-1),
            actions=actions if actions.ndim > 1 else actions.unsqueeze(-1),
            rewards=rewards if rewards.ndim > 1 else rewards.unsqueeze(-1),
        )

        return trajectory

    # Trajectory pair
    def uniform_trajectory_pair(self, traj_length, time_window):
        trajectory1 = self.uniform_trajectory(traj_length, time_window)
        trajectory2 = self.uniform_trajectory(traj_length, time_window)
        return trajectory1, trajectory2

    # Batch of trajectories
    def uniform_trajectory_batch(self, traj_length, time_window, batch_size):
        return [self.uniform_trajectory(traj_length, time_window) for _ in range(batch_size)]

    # Batch of trajectory pairs
    def uniform_trajectory_pair_batch(self, traj_length, time_window, batch_size):
        return [self.uniform_trajectory_pair(traj_length, time_window) for _ in range(batch_size)]

    def sum_rewards(self, traj):
        return traj.rewards.sum().item()

    # TODO Ensemble-based sampling