import numpy as np
import torch
#from numpy.f2py.crackfortran import endifs
from stable_baselines3.common.buffers import ReplayBuffer
from collections import namedtuple

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

    # single trajectory
    def uniform_trajectory(self, traj_length, time_window):
        # just in case, shouldn't really happen
        if self.rb.size() < traj_length or self.rb.size() < time_window or time_window < traj_length:
            raise ValueError("Not enough data to sample")

        TrajectorySamples = namedtuple("TrajectorySamples", ["states", "actions", "rewards"])

        # random start index (exclude end of buffer)
        min_start_index = self.rb.size() - time_window + 1  # only new trajectories since last query
        max_start_index = self.rb.size() - traj_length + 1  # pick trajectories so they finish before end of buffer
        start_index = np.random.randint(min_start_index, max_start_index)
        end_index = start_index + traj_length

        # extract states, actions, rewards
        states = torch.tensor(self.rb.observations[start_index:end_index])
        actions = torch.tensor(self.rb.actions[start_index:end_index])
        rewards = torch.tensor(self.rb.rewards[start_index:end_index])

        # name tensors for better access
        trajectory = TrajectorySamples(
            states=states.squeeze(1),
            actions=actions.squeeze(1),
            rewards=rewards,
        )

        return trajectory
        # TrajectorySamples(states=tensor([[States1], [States2], ..., [States_n]]),
        # actions=tensor([[Actions1], [Actions2], ..., [Actions_n]]),
        # rewards=tensor([[Reward1], [Rewards2], ..., [Reward_n]]))


    # trajectory pair
    def uniform_trajectory_pair(self, traj_length, time_window):
        trajectory1 = self.uniform_trajectory(traj_length, time_window)
        trajectory2 = self.uniform_trajectory(traj_length, time_window)

        return (trajectory1, trajectory2)


    # batch of trajectories
    def uniform_trajectory_batch(self, traj_length, time_window, batch_size):
        trajectories_batch = []

        for _ in range(batch_size):
            trajectory = self.uniform_trajectory(traj_length, time_window)

            trajectories_batch.append(trajectory)

        return trajectories_batch


    # batch of trajectory pairs
    def uniform_trajectory_pair_batch(self, traj_length, time_window, batch_size):
        trajectories_batch = []

        for _ in range(batch_size):
            (trajectory1, trajectory2) = self.uniform_trajectory_pair(traj_length, time_window)

            trajectories_batch.append((trajectory1, trajectory2))

        return trajectories_batch

    # TODO Ensemble-based sampling

    def sum_rewards(self, traj):
        return traj.rewards.sum().item()