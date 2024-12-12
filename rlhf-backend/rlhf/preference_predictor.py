import numpy as np
import torch
from networks import EstimatedRewardNetwork
import torch.optim as optim

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

    def compute_loss(self, sample):
        """
        :param sample: A list of tuples where each tuple contains a pair of trajectories and their corresponding human feedback label.
                       The human feedback label is an indicator of which trajectory is preferred by the human.
        :return: The average entropy loss computed across all trajectory pairs in the sample. This represents the discrepancy
                 between the predicted probabilities and the human-provided feedback.
        """
        entropy_loss = 0
        for trajectory_pair, human_feedback_label in sample:
            predicted_prob = self.compute_predicted_probability(trajectory_pair)
            # human feedback label to tensor conversion for processing
            label_1 = torch.tensor(human_feedback_label, dtype=torch.float)
            label_2 = 1 - label_1

            # calculate loss
            loss_1 = label_1 * torch.log(predicted_prob)
            loss_2 = label_2 * torch.log(1 - predicted_prob)


            entropy_loss += -(loss_1 + loss_2)

        return entropy_loss / len(sample) # Since we calculate the loss for a batch, and not for each trajectory pair # TODO: ask this in the 16.10.2024 meeting

    def train_reward_model(self, reward_network, sample, lr, epochs):
        # Recap: Function compute_predicted_probability, gives us a scalar value for the probability that the human chooses trajectory 0 over trajectory 1
        # Compute_loss takes the human_feedback_label and calculates the entropy loss
        # This function aims to change the weights of the reward model, so that it minimizes the entropy loss
        """
        :param reward_network: The reward model which weights are updated to minimize the entropy loss calculated with compute_loss
        :param sample: A list of tuples where each tuple contains a pair of trajectories and their corresponding human feedback label.
        :param lr: learning rate
        :param epochs: training steps
        :return:
        """
        optimizer = optim.Adam(reward_network.parameters(), lr=lr) #Optimizer used for reward model: Adam

        for epoch in range(epochs):
            #Reset Gradients
            optimizer.zero_grad()

            #Calculate entropy loss
            entropy_loss = reward_network.compute_loss(sample)

            #Backpropagation
            entropy_loss.backward()

            #Update the weights of network
            optimizer.step()







