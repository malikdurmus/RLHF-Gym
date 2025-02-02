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
    """
    A Custom Replaybuffer that is extended with an additional variable: model_rewards
    """
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

    # Override to also add model_rewards
    def add(self, obs, next_obs, action, env_reward, model_reward, done, infos):
        """
        Add the environment observations(obs, next_obs ...) and additionally a model_reward given by our reward network
        """
        super().add(obs, next_obs, action, env_reward, done, infos)
        # return to correct pos (self.pos += 1 in super().add) # TODO handle if rb full
        self.model_rewards[self.pos - 1, :] = model_reward

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

    def relabel(self, reward_models, device):
        """Relabel the ReplayBuffer with rewards given by the average of our reward networks
        :param reward_models: a list with our reward networks
        :param device: "cuda" or "cpu"
        :return: None
        """
        num_entries = self.buffer_size if self.full else self.pos

        for idx in range(num_entries):
            # Extract stored transition
            action = torch.tensor(self.actions[idx], device=device, dtype=torch.float32)
            state = torch.tensor(self.observations[idx], device=device, dtype=torch.float32)

            # Compute the new reward using the reward models' mean
            rewards = []
            with torch.no_grad():
                for reward_model in reward_models:
                    reward = reward_model.forward(action=action, observation=state)
                    rewards.append(reward.cpu().numpy())

            # Compute mean reward
            mean_reward = np.mean(rewards, axis=0)

            self.model_rewards[idx] = mean_reward
