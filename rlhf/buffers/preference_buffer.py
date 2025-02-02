import numpy as np
import torch

from rlhf.buffers.sampler import TrajectorySamples


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

    def __len__(self):
        return len(self.buffer)

    def reset(self):    # not used anymore
        self.buffer.clear()

    def copy(self, apply_tda=True, min_length=None, max_length=None):
        """
        Create a copy of the current PreferenceBuffer object.
        :param apply_tda: If True, apply Temporal Data Augmentation (TDA) to the copied buffer.
        :param min_length: Minimum length of the cropped segments (required if apply_tda is True).
        :param max_length: Maximum length of the cropped segments (required if apply_tda is True).
        :return: A new PreferenceBuffer object with the same (or augmented) contents.
        """
        # Create a new PreferenceBuffer instance with the same buffer size
        new_buffer = PreferenceBuffer(self.buffer_size)

        # Copy the contents of the current buffer to the new buffer
        if apply_tda:
            if min_length is None or max_length is None:
                raise ValueError("min_length and max_length must be provided if apply_tda is True")

            # Apply TDA to each trajectory pair in the buffer
            for trajectories, preference in self.buffer:
                augmented_trajectories = tda((trajectories[0], trajectories[1]), min_length, max_length)
                new_buffer.add(augmented_trajectories, preference)
        else:
            # Copy the buffer without applying TDA
            new_buffer.buffer = self.buffer.copy()

        return new_buffer

    def combine_samples(self, primary_sample):
        """
        Combines a list of TrajectorySamples with a PreferenceBuffer containing TrajectorySamples.

        Args:
            sample (list[TrajectorySamples]): A list of TrajectorySamples.
            augmented_sample (PreferenceBuffer): A PreferenceBuffer object containing TrajectorySamples.

        Returns:
            list[TrajectorySamples]: A combined list of TrajectorySamples.
        """
        # Extract the list of TrajectorySamples from the augmented_sample (PreferenceBuffer)
        augmented_list = self.buffer  # Assuming augmented_sample has a `list` attribute

        # Combine the two lists of TrajectorySamples
        combined_list = primary_sample + augmented_list

        # Might be a good idea for batch processing
        # combined_states = torch.cat([s.states for s in combined_list], dim=0)
        # combined_actions = torch.cat([s.actions for s in combined_list], dim=0)
        # combined_env_rewards = torch.cat([s.env_rewards for s in combined_list], dim=0)

        # Create a new TrajectorySamples object with the combined fields
        # combined_trajectory_samples = TrajectorySamples(
        #     states=combined_states,
        #     actions=combined_actions,
        #     env_rewards=combined_env_rewards
        # )


        return combined_list


# Static
def tda(trajectory_pair, min_length, max_length):
    """
    Apply temporal data augmentation to a pair of trajectory segments.

    :param trajectory_pair: a trajectory pair.
    :param min_length: Minimum length of the cropped segments.
    :param max_length: Maximum length of the cropped segments.
    :return: A pair of cropped segments.
    """

    segment0, segment1 = trajectory_pair

    # Ensure both segments have the same length after cropping
    crop_length = np.random.randint(min_length, max_length + 1)  # H'

    # Calculate the maximum possible start index to ensure the crop_length is valid
    max_start_index = min(len(segment0.states), len(segment1.states)) - crop_length
    start_index = np.random.randint(0, max_start_index + 1)  # max (H - H')

    # Crop all fields of segment0
    cropped_segment0 = TrajectorySamples(
        states=segment0.states[start_index: start_index + crop_length],
        actions=segment0.actions[start_index: start_index + crop_length],
        env_rewards=segment0.env_rewards
    )

    # Crop all fields of segment1
    cropped_segment1 = TrajectorySamples(
        states=segment1.states[start_index: start_index + crop_length],
        actions=segment1.actions[start_index: start_index + crop_length],
        env_rewards=segment1.env_rewards
    )

    return cropped_segment0, cropped_segment1