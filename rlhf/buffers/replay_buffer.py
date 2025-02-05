import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from gym import spaces
from typing import Union, NamedTuple

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    env_rewards: torch.Tensor
    model_rewards: torch.Tensor

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

        # Include storage of model_rewards
        self.model_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Include storge of infos
        self.infos = [None] * self.buffer_size
        # Include Mujoco Internal State
        self.full_states = [None] * self.buffer_size

    # Override to also add model_rewards and infos
    def add(self, obs, next_obs, action, env_reward, model_reward, done, infos, full_state):
        super().add(obs, next_obs, action, env_reward, done, infos)
        self.model_rewards[(self.pos - 1) % self.buffer_size, :] = model_reward
        self.infos[(self.pos - 1) % self.buffer_size] = infos
        self.full_states[(self.pos - 1) % self.buffer_size] = full_state

    # Override to also sample model_rewards
    def sample(self, batch_size, env=None):
        # Generate random indices
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        # Collect data (except model_rewards)
        data = self._get_samples(batch_inds, env)

        # Collect model_rewards
        model_rewards = torch.tensor(self.model_rewards[batch_inds], device=self.device)

        return ReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            env_rewards=data.rewards,
            model_rewards=model_rewards,
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

    def relabel(self, reward_models, device, batch_size):
        num_entries = self.buffer_size if self.full else self.pos

        for start_idx in range(0, num_entries, batch_size):
            end_idx = min(start_idx + batch_size, num_entries)

            # Extract stored transitions
            actions = torch.tensor(self.actions[start_idx:end_idx], device=device, dtype=torch.float32)
            states = torch.tensor(self.observations[start_idx:end_idx], device=device, dtype=torch.float32)

            # Compute the new reward using the reward models' mean
            rewards = []
            with torch.no_grad():
                for reward_model in reward_models:
                    reward = reward_model.forward(action=actions, observation=states)
                    rewards.append(reward.cpu().numpy())

            # Compute mean reward
            mean_reward = np.mean(rewards, axis=0)

            self.model_rewards[start_idx:end_idx] = mean_reward