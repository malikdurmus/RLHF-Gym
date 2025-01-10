import torch
from networks import EstimatedRewardNetwork
import torch.optim as optim

class PreferencePredictor:

    def __init__(self, reward_networks: list, reward_model_lr, device):
        self.reward_networks = reward_networks
        self.device = device
        self.optimizers = [
            optim.Adam(reward_network.parameters(), lr=reward_model_lr) for reward_network in reward_networks
        ]  # Optimizer used for reward models: Adam

    # for each data in sample we will use the predict function. after that we will use the preference to calculate entropy loss
    def compute_predicted_probabilities(self, trajectories):
        """
        :param trajectories: A tuple consisting of two trajectories, where each trajectory is an iterable of tuples. Each tuple represents (action, state) pairs from the trajectory.
        :return: A scalar value representing the predicted probability that the human will choose the first trajectory over the second. Returns None if the trajectories have mismatched lengths.
        """
        # trajectories is a tuple that consist of two trajectories (torch tensors)
        # Here I consider a trajectory to be an iterable type of tuples like: [(action1,state1),(action3,state2), (action0,state3) ...]
        # We can change this later on
        trajectory0, trajectory1 = trajectories[0], trajectories[1]

        states0, actions0 = trajectory0.states, trajectory0.actions
        states1, actions1 = trajectory1.states, trajectory1.actions

        predictions = []

        for model_idx, reward_model in enumerate(self.reward_networks):
            prob0, prob1 = 0.0, 0.0
            for state, action in zip(states0, actions0):
                action = action.to(self.device)
                state = state.to(self.device)
                prob0 += reward_model(action=action, observation=state)
            for state, action in zip(states1, actions1):
                action = action.to(self.device)
                state = state.to(self.device)
                prob1 += reward_model(action=action, observation=state)

            # predicted_prob = torch.exp(total_prob0) / (torch.exp(total_prob0) + torch.exp(total_prob1)) #probability, that the human will chose trajectory0 over trajectory1
            max_prob = torch.max(prob0, prob1)
            predicted_prob = torch.exp(prob0 - max_prob) / (
                torch.exp(prob0 - max_prob) + torch.exp(prob1 - max_prob)
            )

            predictions.append((model_idx, predicted_prob))

        return predictions

    def _compute_losses(self, sample):
        """
        :param sample: A list of tuples where each tuple contains a pair of trajectories and their corresponding human feedback label.
                       The human feedback label is an indicator of which trajectory is preferred by the human.
        :return: The average entropy loss computed across all trajectory pairs in the sample. This represents the discrepancy
                 between the predicted probabilities and the human-provided feedback.
        """
        model_losses = [0.0 for _ in range(len(self.reward_networks))]

        for trajectory_pair, human_feedback_label in sample:
            predicted_probs = self.compute_predicted_probabilities(trajectory_pair)
            for idx, predicted_prob in predicted_probs:
                # human feedback label to tensor conversion for processing
                label_1 = torch.tensor(human_feedback_label[0], dtype=torch.float, device=self.device)
                label_2 = 1 - label_1

                # calculate loss for this model
                loss_1 = label_1 * torch.log(predicted_prob+1e-6)
                loss_2 = label_2 * torch.log(1 - predicted_prob + 1e-6)
                model_losses[idx] += -(loss_1 + loss_2)

        return model_losses #loss for whole batch

    def train_reward_model(self, sample):
        # Recap: Function compute_predicted_probability, gives us a scalar value for the probability that the human chooses trajectory 0 over trajectory 1
        # Compute_loss takes the human_feedback_label and calculates the entropy loss
        # This function aims to change the weights of the reward model, so that it minimizes the entropy loss
        """
        :param sample: A list of tuples where each tuple contains a pair of trajectories and their corresponding human feedback label.
        :return: None
        """

        model_losses = self._compute_losses(sample)

        for i, (reward_model, optimizer) in enumerate(zip(self.reward_networks, self.optimizers)):
            # Reset Gradients
            optimizer.zero_grad()

            # Backpropagation
            model_losses[i].backward()

            # Update the weights of network
            optimizer.step()

        entropy_loss = sum(model_losses) / len(model_losses)

        return entropy_loss