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

    # Single trajectory
    def uniform_trajectory(self, traj_length, time_window, synthetic_feedback):
        if self.rb.size() < traj_length or self.rb.size() < time_window or time_window < traj_length:
            raise ValueError("Not enough data to sample, consider adjusting args")

        # Random start index (exclude end of buffer)
        min_start_index = self.rb.size() - time_window + 1
        max_start_index = self.rb.size() - traj_length + 1
        start_index = np.random.randint(min_start_index, max_start_index)
        end_index = start_index + traj_length

        # extract states, actions, env_rewards
        states = torch.tensor(self.rb.observations[start_index:end_index])
        actions = torch.tensor(self.rb.actions[start_index:end_index])

        if synthetic_feedback:
            env_rewards = torch.tensor(self.rb.rewards[start_index:end_index])
            env_rewards = env_rewards if env_rewards.ndim > 1 else env_rewards.unsqueeze(-1)
        else:
            env_rewards = None

        # name tensors for better access
        trajectory = TrajectorySamples(
            states=states if states.ndim > 1 else states.unsqueeze(-1),
            actions=actions if actions.ndim > 1 else actions.unsqueeze(-1),
            env_rewards=env_rewards,
        )

        trajectory.to(device=self.device) #since this is an immutable tuple, the tensors have to be recreated everytime,
                                            # which is not good TODO: better doc here dont commit, SOLVED, see branch fix/video-fix

        return trajectory
        # TrajectorySamples(states=tensor([[States1], [States2], ..., [States_n]]),
        # actions=tensor([[Actions1], [Actions2], ..., [Actions_n]]),
        # env_rewards=tensor([[Reward1], [Reward2], ..., [Reward_n]]))


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

            # predictions already a tensor, calculate variance with predictions.var()
            # convert result back to python value with tensor.item()
            variance = predictions.var().item()

            variance_list.append((traj_pair, variance))

            """
            predicted_prob_list = []
            for predicted_prob  in predictions:
                # TODO maybe keep calculations on the gpu with tensor.var() (Thang Long)
                predicted_prob = predicted_prob.detach().cpu().numpy()
                # append variance to a list
                predicted_prob_list.append(predicted_prob)

            # Calculate the variance
            variance_list.append((traj_pair, np.var(predicted_prob_list)))
            """


        # sort list in descending order
        sorted_variance = sorted(variance_list, key=lambda x: x[1], reverse=True)

        return [element[0] for element in sorted_variance[:ensemble_size]]