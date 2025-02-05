import numpy as np

class IntrinsicRewardCalculator:
    def __init__(self, k):
        self.states = []
        self.k = k

    def add_state(self, state):
        self.states.append(state)

    def compute_intrinsic_reward(self, state):
        # Convert to NumPy-Array
        states_array = np.vstack(self.states)

        # Calculate distances via Euclidian Distance
        distances = np.linalg.norm(states_array - state, axis=1)

        # Get the k smallest distances
        k_distances = np.sort(distances)[:self.k]

        # Get the k smallest distance
        k_distance = k_distances[-1] if len(k_distances) == self.k else k_distances[-1]

        # Intrinsic reward
        reward = np.log(k_distance + 1e-6)

        return reward

    def __len__(self):
        return len(self.states)