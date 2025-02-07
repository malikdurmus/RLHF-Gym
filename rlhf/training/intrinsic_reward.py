import numpy as np
from rlhf.args import Args

class IntrinsicRewardCalculator:
    """
    A class for calculating intrinsic rewards based on the novelty of states.

    This class maintains a history of states and computes intrinsic rewards for new states
    based on their distance to the k-nearest neighbors in the history.

    :param k: The number of nearest neighbors to consider for intrinsic reward calculation.
    """
    def __init__(self, k):
        """
        Initializes the IntrinsicRewardCalculator with the specified number of nearest neighbors.
        :param k: The number of nearest neighbors to consider for intrinsic reward calculation.
        """
        self.states = []
        self.k = k

        # values for std. deviation calculation
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def add_state(self, state):
        """
        Adds a new state to the history of states.

        :param state: The state to add to the history.
        """
        self.states.append(state)

    def update(self, new_value):
        """
        Updates the values for the running std.Deviation estimate

        :param new_value: new intrinsic reward
        """
        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    def std_deviation(self):
        """
        Calculates the current standard deviation of all intrinsic rewards

        :return: Current std. deviation
        """
        if self.n < 2:
            return 1
        else:
            return (self.M2 / self.n) ** 0.5

    def compute_intrinsic_reward(self, state):
        """
        Computes the intrinsic reward for a given state based on its novelty.

        The novelty is determined by the distance to the k-nearest neighbors in the history of states.
        The intrinsic reward is calculated as the logarithm of the k-th smallest distance.

        :param state: The state for which to compute the intrinsic reward.
        :return: The intrinsic reward for the given state.
        """
        # Convert to NumPy-Array
        states_array = np.vstack(self.states)

        # Calculate distances via Euclidian Distance
        distances = np.linalg.norm(states_array - state, axis=1)

        # Get the k smallest distances
        k_distances = np.sort(distances)[:self.k]

        # Get the k smallest distance
        k_distance = k_distances[-1]

        # Intrinsic reward
        reward = np.log(k_distance + 1e-6)

        if Args.normalize_int_rewards:
            # Update std. deviation values
            self.update(reward)
            # Normalize reward by dividing by standard devíation
            reward = reward/self.std_deviation()

        return reward

    def __len__(self):
        """
        Returns the number of states currently stored in the history.

        :return: The number of states in the history.
        """
        return len(self.states)