import numpy as np

from rlhf.buffers.sampler import TrajectorySamples

class PreferenceBuffer:
    """
    This class creates a Preference Buffer which saves our trajectory pair and the corresponding preference.
    """
    def __init__(self, buffer_size):
        """
        Initialize the buffer.
        :param buffer_size: a specific size of our buffer.
        """
        self.buffer = []
        self.buffer_size = buffer_size
        self.pos = 0

    def add(self, trajectories, preference):
        """
        Adding samples to our Preference Buffer.
        :param trajectories: a trajectory pair.
        :param preference: a preference.
        """
        if len(trajectories) != 2:
            raise Exception("More than 2 trajectories")
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)

        self.buffer.append([trajectories, preference])
        self.pos += 1


    def sample_with_validation_sample(self, batch_size, recent_data_size, replace):
        """
        Samples a batch from the buffer with a train-validation split.
        :param batch_size: The amount of samples.
        :param recent_data_size: The number of elements considered for validation (from the end of the buffer).
        :param replace: A boolean whether we sample the same samples multiple times.

        :return: Two lists: A Training subset and Validation subset of the sampled batch.
        """


        # Get all valid validation indices (since last feedback query)
        val_start_idx = max(0, self.pos - recent_data_size)
        val_candidate_indices = np.arange(val_start_idx, self.pos)

        # Sample 1/e of last feedback, at least 1
        val_sample_size = max(1, int(recent_data_size / np.e))
        val_indices = np.random.choice(val_candidate_indices,
                                              size=min(val_sample_size, len(val_candidate_indices)), replace=False)

        # Get all valid training indices
        remaining_indices = np.setdiff1d(np.arange(self.pos), val_indices)

        # Sample random training indices
        train_indices = np.random.choice(remaining_indices, size=min(batch_size, len(remaining_indices)),
                                                replace=replace)

        # Extract samples
        train_samples = [self.buffer[i] for i in train_indices]
        val_samples = [self.buffer[i] for i in val_indices]

        return train_samples, val_samples

    def sample(self, batch_size, replace=False):
        """
        Samples a batch from the Preference Buffer.
        :param batch_size: The amount of samples.
        :param replace: A boolean whether we sample the same samples multiple times, Default: False.

        :return: A list of sampled elements that are saved in the buffer.
        """
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=replace)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        """
        :return: length of our buffer.
        """
        return len(self.buffer)

    def reset(self):
        """
        Empty the buffer .
        """
        self.buffer.clear()

    def copy(self, apply_tda=True,crop_size=None):
        """
        Create a copy of the current PreferenceBuffer object.
        :param apply_tda: If True, apply Temporal Data Augmentation (TDA) to the copied buffer.
        :param crop_size: cropping intensity for the segments (required if apply_tda is True).

        :return: A new PreferenceBuffer object with the same (or augmented) contents.
        """
        # Create a new PreferenceBuffer instance with the same buffer size
        new_buffer = PreferenceBuffer(self.buffer_size)

        # Copy the contents of the current buffer to the new buffer
        if apply_tda:
            if crop_size is None:
                raise ValueError("crop must be provided if apply_tda is True")

            # Apply TDA to each trajectory pair in the buffer
            for trajectories, preference in self.buffer:
                augmented_trajectories = tda((trajectories[0], trajectories[1]), crop_size)
                new_buffer.add(augmented_trajectories, preference)
        else:
            # Copy the buffer without applying TDA
            new_buffer.buffer = self.buffer.copy()

        return new_buffer

    def combine_samples(self, primary_sample):
        """
        Combines a list of TrajectorySamples with a PreferenceBuffer containing TrajectorySamples.
        :param primary_sample: (list[TrajectorySamples]): A list of TrajectorySamples.

        :return: list[TrajectorySamples]: A combined list of TrajectorySamples.
        """
        # Extract the list of TrajectorySamples from the augmented_sample (PreferenceBuffer)
        augmented_list = self.get_buffer()  # Assuming augmented_sample has a `list` attribute

        # Combine the two lists of TrajectorySamples
        combined_list = primary_sample + augmented_list

        return combined_list

    def get_buffer(self):
        return self.buffer


# Static
def tda(trajectory_pair,crop_size):
    """
    Apply temporal data augmentation to a pair of trajectory segments.
    :param crop_size: crop size for the segments
    :param trajectory_pair: a trajectory pair.

    :return: A pair of cropped segments.
    :raises ValueError: If crop size is too large for the given trajectory length.
    """
    segment0, segment1 = trajectory_pair

    # Ensure segments have same length before cropping
    if len(segment0.states) != len(segment1.states):
        raise Exception("Trajectory lengths differ")

    segment_length = len(segment0.states)

    # Ensure crop size is not too large
    if crop_size * 2 >= segment_length:
        raise ValueError(f"Crop size {crop_size} is too large for trajectory length {segment_length}. "
                         f"Must be smaller than {segment_length // 2}.")

    max_length = segment_length - crop_size
    min_length = segment_length - (crop_size * 2)

    # Ensure both segments have the same length after cropping
    crop_length = np.random.randint(min_length, max_length + 1)  # H'

    # Calculate the maximum possible start index to ensure the crop_length is valid
    max_start_index = segment_length - crop_length
    start_index = np.random.randint(0, max_start_index + 1)  # max (H - H')

    # Crop all fields of segment 0 and segment 1
    cropped_segment0 = TrajectorySamples(
        states=segment0.states[start_index: start_index + crop_length],
        actions=segment0.actions[start_index: start_index + crop_length],
        env_rewards=segment0.env_rewards[start_index: start_index + crop_length],
        infos=segment0.infos[start_index: start_index + crop_length],
        full_states=segment0.full_states[start_index: start_index + crop_length]
    )

    cropped_segment1 = TrajectorySamples(
        states=segment1.states[start_index: start_index + crop_length],
        actions=segment1.actions[start_index: start_index + crop_length],
        env_rewards=segment1.env_rewards[start_index: start_index + crop_length],
        infos=segment1.infos[start_index: start_index + crop_length],
        full_states=segment1.full_states[start_index: start_index + crop_length]
    )

    return cropped_segment0, cropped_segment1