import numpy as np
import torch
from networks import EstimatedRewardNetwork

class PreferencePredictor:

    def __init__(self, reward_network: EstimatedRewardNetwork):
        self.reward_network = reward_network

    # for each data in sample we will use the predict function. after that we will use the preference to calculate entropy loss
    def compute_predicted_probability(self, trajectories):
        """
        :param trajectories: A tuple consisting of two trajectories, where each trajectory is an iterable of tuples. Each tuple represents (action, state) pairs from the trajectory.
        :return: A scalar value representing the predicted probability that the human will choose the first trajectory over the second. Returns None if the trajectories have mismatched lengths.
        """
        # trajectories is a tuple that consist of two trajectories (torch tensors)
        # Here I consider a trajectory to be an iterable type of tuples like: [(action1,state1),(action3,state2), (action0,state3) ...]
        # We can change this later on
        trajectory0, trajectory1 = trajectories[0], trajectories[1]

        if trajectory0.length != trajectory1.length:
            print("trajectory_lengths do not match")
            return None
        total_prob0 = 0
        total_prob1 = 0
        for action, state in trajectory0:
            prob_for_action = self.reward_network(action=torch.tensor(action),
                                                  observation=torch.tensor(state)) # estimated probability, that the human will prefer action 0
            total_prob0 += prob_for_action
        for action, state in trajectory1:
            prob_for_action = self.reward_network(action=torch.tensor(action),
                                                  observation=torch.tensor(state))  # estimated probability, that the human will prefer action 1
            total_prob1 += prob_for_action

        predicted_prob = np.exp(total_prob0) / (np.exp(total_prob0) + np.exp(total_prob1)) #probability, that the human will chose trajectory0 over trajectory1
        return predicted_prob

    def compute_loss(self):


        pass

