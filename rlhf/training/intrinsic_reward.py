import numpy as np
from rlhf.args import Args

class IntrinsicRewardCalculator:
    def __init__(self, k):
        self.states = []
        self.k = k

        # values for std. deviation calculation
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def add_state(self, state):
        self.states.append(state)

    def update(self, new_value):
        """
        Updates the values for the running std.Deviation estimate

        :param new_value: newest k_distance
        """
        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    def std_deviation(self):
        """
        Calculates the current standard deviation of k_distance values

        :return: Current std. deviation
        """
        if self.n < 2:
            return 1
        else:
            return (self.M2 / self.n) ** 0.5

    def compute_intrinsic_reward(self, state):
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
        return len(self.states)