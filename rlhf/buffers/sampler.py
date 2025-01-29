import numpy as np
import torch
from dataclasses import dataclass

### TRAJECTORY SAMPLER ###
@dataclass
class TrajectorySamples:
    states: torch.Tensor
    actions: torch.Tensor
    env_rewards: torch.Tensor

class TrajectorySampler:
    def __init__(self, rb, device):
        self.rb = rb
        self.device = device

    def uniform_trajectory(self, traj_length, time_window, synthetic_feedback):
        """
        Samples a uniformly random trajectory from the replay buffer.
        :param traj_length: Length of the trajectory in steps. Must be less than or equal to the current buffer size.
        :param time_window: Relevant range within the replay buffer to sample from. Defines the range between
                               the most recent and earlier entries. Must be greater than `traj_length`.
        :param synthetic_feedback: Flag to indicate whether to include synthetic feedback.
                                       If True, includes `env_rewards` in the trajectory; otherwise excludes them.
        :return: TrajectorySamples: A named tuple containing the sampled trajectory:
                - `states` (torch.Tensor): States of the trajectory, shape `(traj_length, state_dim)`.
                - `actions` (torch.Tensor): Actions of the trajectory, shape `(traj_length, action_dim)`.
                - `env_rewards` (torch.Tensor or None): Rewards of the trajectory if `synthetic_feedback` is True, otherwise None.
        :raises: ValueError: If the buffer size or time window is insufficient to sample the requested trajectory length.
        """

        if self.rb.size() < traj_length or time_window < traj_length:
            raise ValueError("Not enough data to sample, consider adjusting args")

        # random start index
        min_start_index = (self.rb.pos - time_window) % self.rb.size()
        max_start_index = (self.rb.pos - traj_length) % self.rb.size()
        start_index = np.random.randint(min_start_index, max_start_index)
        end_index = start_index + traj_length

        # extract states, actions, env_rewards
        states = torch.tensor(self.rb.observations[start_index:end_index], device=self.device)
        actions = torch.tensor(self.rb.actions[start_index:end_index], device=self.device)

        if synthetic_feedback:
            env_rewards = torch.tensor(self.rb.rewards[start_index:end_index], device=self.device)
            env_rewards = env_rewards if env_rewards.ndim > 1 else env_rewards.unsqueeze(-1)
        else:
            env_rewards = None

        # name tensors for better access
        trajectory = TrajectorySamples(
            states=states if states.ndim > 1 else states.unsqueeze(-1),
            actions=actions if actions.ndim > 1 else actions.unsqueeze(-1),
            env_rewards=env_rewards,
        )

        return trajectory


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

    def ensemble_sampling(self,ensemble_size, uniform_size, traj_length, time_window, synthetic_feedback, preference_optimizer):
        """
        Performs ensemble-based sampling by evaluating variance across an ensemble of predicted probabilities.

        :param ensemble_size: Number of trajectory pairs to select with the highest variance.
        :param uniform_size: Number of trajectory pairs to sample uniformly before selecting the top pairs.
        :param traj_length: Length of the trajectory in steps. Must be less than or equal to the current buffer size.
        :param time_window: Relevant range within the replay buffer to sample from. Defines the range between
                            the most recent and earlier entries. Must be greater than `traj_length`.
        :param synthetic_feedback: Flag to indicate whether to include synthetic feedback.
                                    If True, includes `env_rewards` in the trajectory; otherwise excludes them.
        :param preference_optimizer: A preference optimizer object that computes predicted probabilities
                                    for each trajectory pair using an ensemble of reward models.
        :return: A list of `TrajectorySamples` pairs, sorted by their ensemble variance in descending order.
                  Contains up to `ensemble_size` pairs.
        :raises: ValueError: If the buffer size or time window is insufficient to sample the requested trajectory length.
        """

        # Create empty list for variance: ((traj1, traj2), variance)
        variance_list = []
        for _ in range(uniform_size):
            # sample one trajectory pair
            traj_pair = self.uniform_trajectory_pair(traj_length, time_window, synthetic_feedback)

            # pass traj to each network: for reward_model in reward_networks, networks calculate a reward
            predictions = preference_optimizer.compute_predicted_probabilities(traj_pair)

            predicted_prob_list = []
            for predicted_prob  in predictions:
                # TODO maybe keep calculations on the gpu with tensor.var() (Thang Long)
                predicted_prob = predicted_prob.detach().cpu().numpy()
                # append variance to a list
                predicted_prob_list.append(predicted_prob)

            # Calculate the variance
            variance_list.append((traj_pair, np.var(predicted_prob_list)))

        # sort list in descending order
        sorted_variance = sorted(variance_list, key=lambda x: x[1], reverse=True)

        return [element[0] for element in sorted_variance[:ensemble_size]]