import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from collections import namedtuple
from gym import spaces
from typing import Union, NamedTuple


# Override to include true_rewards
class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    true_rewards: torch.Tensor

class CustomReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[torch.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        # Initialize SB3 ReplayBuffer
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage,
                         handle_timeout_termination)

        # Include storage of true_rewards
        self.true_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    # Override to also add true_rewards
    def add(self, obs, next_obs, action, reward, true_reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
        self.true_rewards[self.pos] = np.array(true_reward)

    # Override to also sample true_rewards
    def sample(self, batch_size, env=None):
        # Generate random indices
        if self.full:
            batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        # Collect data (except true_rewards)
        data = self._get_samples(batch_inds, env)

        # Collect true_rewards
        true_rewards = torch.tensor(self.true_rewards[batch_inds], device=self.device)

        return ReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            rewards=data.rewards,
            true_rewards=true_rewards,
        )

    @classmethod
    def initialize(cls, envs, buffer_size, device):
        envs.single_observation_space.dtype = np.float32
        return cls(
            buffer_size=buffer_size,
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            device=device,
            handle_timeout_termination=False,
        )


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

    def reset(self):
        self.buffer.clear()


class TrajectorySampler:
    def __init__(self, rb):
        self.rb = rb

    # single trajectory
    def uniform_trajectory(self, traj_length, time_window, feedback_mode):
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

        if feedback_mode == "synthetic":
            rewards = torch.tensor(self.rb.rewards[start_index:end_index])
            rewards = rewards if rewards.ndim > 1 else rewards.unsqueeze(-1)
        else:
            rewards = None

        # name tensors for better access
        trajectory = TrajectorySamples(
            states=states if states.ndim > 1 else states.unsqueeze(-1),
            actions=actions if actions.ndim > 1 else actions.unsqueeze(-1),
            rewards=rewards,
        )

        return trajectory
        # TrajectorySamples(states=tensor([[States1], [States2], ..., [States_n]]),
        # actions=tensor([[Actions1], [Actions2], ..., [Actions_n]]),
        # rewards=tensor([[Reward1], [Reward2], ..., [Reward_n]]))


    # trajectory pair
    def uniform_trajectory_pair(self, traj_length, time_window, feedback_mode):
        trajectory1 = self.uniform_trajectory(traj_length, time_window, feedback_mode)
        trajectory2 = self.uniform_trajectory(traj_length, time_window, feedback_mode)

        return (trajectory1, trajectory2)


    # batch of trajectories
    def uniform_trajectory_batch(self, traj_length, time_window, batch_size, feedback_mode):
        trajectories_batch = []

        for _ in range(batch_size):
            trajectory = self.uniform_trajectory(traj_length, time_window, feedback_mode)

            trajectories_batch.append(trajectory)

        return trajectories_batch


    # batch of trajectory pairs
    def uniform_trajectory_pair_batch(self, traj_length, time_window, batch_size, feedback_mode):
        trajectories_batch = []

        for _ in range(batch_size):
            (trajectory1, trajectory2) = self.uniform_trajectory_pair(traj_length, time_window, feedback_mode)

            trajectories_batch.append((trajectory1, trajectory2))

        return trajectories_batch

    # TODO Ensemble-based sampling

    def sum_rewards(self, traj):
        return traj.rewards.sum().item()


def relabel_replay_buffer(rb, reward_models, device):
    num_entries = rb.buffer_size if rb.full else rb.pos

    for idx in range(num_entries):
        # Extract stored transition
        action = torch.tensor(rb.actions[idx], device=device, dtype=torch.float32)
        state = torch.tensor(rb.observations[idx], device=device, dtype=torch.float32)

        # Compute the new reward using the reward models' mean
        rewards = []
        with torch.no_grad():
            for reward_model in reward_models:
                reward = reward_model.forward(action=action, observation=state)
                rewards.append(reward.cpu().numpy())

        # Compute mean reward
        mean_reward = np.mean(rewards, axis=0)

        rb.true_rewards[idx] = mean_reward