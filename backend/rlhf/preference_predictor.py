import torch
from backend.rlhf.networks import EstimatedRewardNetwork
import torch.optim as optim

class PreferencePredictor:

    def __init__(self, reward_network: "EstimatedRewardNetwork", reward_model_lr,device):
        self.reward_network = reward_network
        self.reward_model_lr = reward_model_lr
        self.device = device
        self.optimizer = optim.Adam(self.reward_network.parameters(), lr=reward_model_lr)  # Optimizer used for reward model: Adam

    def train_reward_model(self, sample):
        # Recap: Function compute_predicted_probability, gives us a scalar value for the probability that the human chooses trajectory 0 over trajectory 1
        # Compute_loss takes the human_feedback_label and calculates the entropy loss
        # This function aims to change the weights of the reward model, so that it minimizes the entropy loss
        """
        :param sample: A list of tuples where each tuple contains a pair of trajectories and their corresponding human feedback label.
        :return: None
        """

        # Reset Gradients
        self.optimizer.zero_grad()

        # Calculate entropy loss
        entropy_loss = self._compute_loss(sample)
        entropy_loss2 = self._compute_loss_batch(sample)
        print("entropy_loss2: ",entropy_loss2)
        # Backpropagation
        entropy_loss.backward()

        # Update the weights of network
        self.optimizer.step()

        return entropy_loss




    # for each data in sample we will use the predict function. after that we will use the preference to calculate entropy loss
    def _compute_predicted_probability(self, trajectories):
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

        if len(states0) != len(states1) or len(actions0) != len(actions1):
            raise ValueError("Trajectory lengths do not match")

        total_prob0 = 0
        total_prob1 = 0
        for state, action in zip(states0, actions0):
            action = action.to(self.device)
            state = state.to(self.device)
            prob_for_action = self.reward_network(action=action,
                                                  observation=state) # estimated probability, that the human will prefer action 0

            total_prob0 += prob_for_action
        for state, action in zip(states1, actions1):
            action = action.to(self.device)
            state = state.to(self.device)
            prob_for_action = self.reward_network(action=action,
                                                  observation=state)  # estimated probability, that the human will prefer action 1
            total_prob1 += prob_for_action # Tensor of shape {Tensor : {1,1}} , tested

        max_prob = torch.max(total_prob0, total_prob1)
        predicted_prob = torch.exp(total_prob0 - max_prob) / (
                torch.exp(total_prob0 - max_prob) + torch.exp(total_prob1 - max_prob)
        )
        return predicted_prob
    # synthetic feedbacks needs to be changed, it gives
    def _compute_loss(self, sample):
        """
        :param sample: A list of lists where each inner list contains:
               - A tuple (trajectory1, trajectory2) (both trajectories)
               - An integer representing human feedback (0 or 1)
        :return: The cumulative entropy loss computed across all trajectory pairs in the sample.
        """
        entropy_loss = torch.tensor(0.0, requires_grad=True)
        for trajectory_pair, human_feedback_label in sample:
            predicted_prob = self._compute_predicted_probability(trajectory_pair)
            # human feedback label to tensor conversion for processing
            label_1 = torch.tensor(human_feedback_label, dtype=torch.float)
            label_1 = label_1.to(self.device)

            predicted_prob = torch.clamp(predicted_prob, min=1e-7, max=1 - 1e-7)

            # calculate loss
            loss_1 = label_1 * torch.log(predicted_prob)
            loss_2 = (1 - label_1) * torch.log(1 - predicted_prob)

            entropy_loss = entropy_loss + -(loss_1 + loss_2 )
        res = entropy_loss
        return entropy_loss #loss for whole batch

    def _compute_loss_batch(self, sample):
        """
        Compute the cumulative entropy loss for the given sample.
        """
        entropy_loss = torch.tensor(0.0, requires_grad=True)


        for trajectory_pair, human_feedback_label in sample:
            predicted_prob = self._compute_predicted_probability_batch(trajectory_pair)

            # label to tensor
            label = torch.tensor(human_feedback_label, dtype=torch.float, device=self.device)

            # clamp to to avoid log(0)
            predicted_prob = torch.clamp(predicted_prob, min=1e-7, max=1 - 1e-7)

            # Compute binary cross-entropy loss
            loss = -label * torch.log(predicted_prob) - (1 - label) * torch.log(1 - predicted_prob)
            entropy_loss = entropy_loss + loss

        return entropy_loss

    def _compute_predicted_probability_batch(self, trajectories):
        """
        Compute the predicted probability of human preference for trajectory0 over trajectory1.
        """
        trajectory0, trajectory1 = trajectories

        # Ensure both trajectories have the same length
        if trajectory0.states.size(0) != trajectory1.states.size(0):
            raise ValueError("Trajectory lengths do not match.")

        # Move data to the correct device
        trajectory0.to(self.device)
        trajectory1.to(self.device)

        # Batch process all state-action pairs
        rewards0 = self.reward_network(trajectory0.actions, trajectory0.states)  # Shape: [N, 1]
        rewards1 = self.reward_network(trajectory1.actions, trajectory1.states)  # Shape: [N, 1]

        # Sum rewards for each trajectory
        total_reward0 = rewards0.sum()  # Scalar
        total_reward1 = rewards1.sum()  # Scalar

        # Stabilize softmax calculation
        max_reward = torch.max(total_reward0, total_reward1)
        predicted_prob = torch.exp(total_reward0 - max_reward) / (
                torch.exp(total_reward0 - max_reward) + torch.exp(total_reward1 - max_reward)
        )

        return predicted_prob
