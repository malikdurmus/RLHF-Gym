import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from collections import namedtuple
from gym import spaces
from dataclasses import dataclass
from typing import Union, NamedTuple

@dataclass
class TrajectorySamples:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    def to(self, device: torch.device):
        """Move all tensors to the given device."""
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        if self.rewards is not None:
            self.rewards = self.rewards.to(device)

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

    # Override to also sample true_rewards
    def sample(self, batch_size, env=None):
        # Generate random indices
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
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
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, trajectories, preference):
        if len(trajectories) != 2:
            raise Exception("More than 2 trajectories")
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append([trajectories, preference])

    def sample(self, batch_size, replace=False):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=replace)
        return [self.buffer[i] for i in indices]

    def reset(self):
        self.buffer.clear()


class TrajectorySampler:
    def __init__(self, rb, device):
        self.rb = rb
        self.device = device

    # Single trajectory
    def uniform_trajectory(self, traj_length, time_window, synthetic_feedback):
        if self.rb.size() < traj_length or self.rb.size() < time_window or time_window < traj_length:
            raise ValueError("Not enough data to sample, consider adjusting args")

        # Random start index (exclude end of buffer)
        min_start_index = self.rb.size() - time_window + 1
        max_start_index = self.rb.size() - traj_length + 1
        start_index = np.random.randint(min_start_index, max_start_index)
        end_index = start_index + traj_length

        # extract states, actions, rewards
        states = torch.tensor(self.rb.observations[start_index:end_index])
        actions = torch.tensor(self.rb.actions[start_index:end_index])

        if synthetic_feedback:
            rewards = torch.tensor(self.rb.rewards[start_index:end_index])
            rewards = rewards if rewards.ndim > 1 else rewards.unsqueeze(-1)
        else:
            rewards = None

        # name tensors for better access
        trajectory = TrajectorySamples(
            states= states if states.ndim > 1 else states.unsqueeze(-1),
            actions=actions if actions.ndim > 1 else actions.unsqueeze(-1),
            rewards=rewards,
        )

        trajectory.to(device=self.device) #since this is an immutable tuple, the tensors have to be recreated everytime, which is not good TODO: better doc herer dont commit

        return trajectory
        # TrajectorySamples(states=tensor([[States1], [States2], ..., [States_n]]),
        # actions=tensor([[Actions1], [Actions2], ..., [Actions_n]]),
        # rewards=tensor([[Reward1], [Reward2], ..., [Reward_n]]))


    # trajectory pair
    def uniform_trajectory_pair(self, traj_length, time_window, synthetic_feedback):
        trajectory1 = self.uniform_trajectory(traj_length, time_window, synthetic_feedback)
        trajectory2 = self.uniform_trajectory(traj_length, time_window, synthetic_feedback)

        return trajectory1, trajectory2


    # batch of trajectories
    def uniform_trajectory_batch(self, traj_length, time_window, batch_size, synthetic_feedback):
        trajectories_batch = []

        for _ in range(batch_size):
            trajectory = self.uniform_trajectory(traj_length, time_window, synthetic_feedback)

            trajectories_batch.append(trajectory)

        return trajectories_batch


    # batch of trajectory pairs
    def uniform_trajectory_pair_batch(self, traj_length, time_window, batch_size, synthetic_feedback):
        trajectories_batch = []

        for _ in range(batch_size):
            (trajectory1, trajectory2) = self.uniform_trajectory_pair(traj_length, time_window, synthetic_feedback)

            trajectories_batch.append((trajectory1, trajectory2))

        return trajectories_batch

    # ensemble-based sampling
    def ensemble_sampling(self,ensemble_size, uniform_size, traj_length, time_window, synthetic_feedback, preference_optimizer):
        # Create empty list for variance: ((traj1, traj2), variance)
        variance_list = []
        for _ in range(uniform_size):
            # sample one trajectory pair
            traj_pair = self.uniform_trajectory_pair(traj_length, time_window, synthetic_feedback)

            # pass traj to each network: for reward_model in reward_networks, networks calculate a reward
            predictions = preference_optimizer.compute_predicted_probabilities(traj_pair)

            predicted_prob_list = []
            for predicted_prob  in predictions:
                # TODO maybe keep calculations on the gpu with tensor.var()
                predicted_prob = predicted_prob.detach().cpu().numpy()
                # append variance to a list
                predicted_prob_list.append(predicted_prob)

            # Calculate the variance
            variance_list.append((traj_pair, np.var(predicted_prob_list)))

        # sort list in descending order
        sorted_variance = sorted(variance_list, key=lambda x: x[1], reverse=True)

        return [element[0] for element in sorted_variance[:ensemble_size]]


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