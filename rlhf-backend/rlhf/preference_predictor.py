import torch
import torch.optim as optim

class PreferencePredictor:

    def __init__(self, reward_networks: list, reward_model_lr, device, l2):
        self.reward_networks = reward_networks
        self.device = device
        self.l2 = l2
        self.reward_model_lr = reward_model_lr
        self.optimizers = self._initialize_optimizers()

    def _initialize_optimizers(self):
        return [
            optim.Adam(
                reward_network.parameters(),
                lr=self.reward_model_lr,
                weight_decay=self.l2
            )
            for reward_network in self.reward_networks
        ]

    def _update_optimizers(self):
        self.optimizers = self._initialize_optimizers()

    # for each data in sample we will use the predict function. after that we will use the preference to calculate entropy loss
    def compute_predicted_probability(self, reward_model, trajectories):
        """
        :param trajectories: A tuple consisting of two trajectories, where each trajectory is an iterable of tuples. Each tuple represents (action, state) pairs from the trajectory.
        :return: A scalar value representing the predicted probability that the human will choose the first trajectory over the second. Returns None if the trajectories have mismatched lengths.
        """
        # trajectories is a tuple that consist of two trajectories (torch tensors)
        # Here I consider a trajectory to be an iterable type of tuples like: [(action1,state1),(action3,state2), (action0,state3) ...]
        # We can change this later on
        trajectory0, trajectory1 = trajectories[0], trajectories[1]

        states0, actions0 = trajectory0.states.to(self.device), trajectory0.actions.to(self.device)
        states1, actions1 = trajectory1.states.to(self.device), trajectory1.actions.to(self.device)

        total_prob0 = 0
        total_prob1 = 0
        for state, action in zip(states0, actions0):
            action = action.to(self.device)
            state = state.to(self.device)
            prob_for_action = reward_model(action=action,
                                                  observation=state) # estimated probability, that the human will prefer action 0
            total_prob0 += prob_for_action
        for state, action in zip(states1, actions1):
            action = action.to(self.device)
            state = state.to(self.device)
            prob_for_action = reward_model(action=action,
                                                  observation=state)  # estimated probability, that the human will prefer action 1
            total_prob1 += prob_for_action

        #predicted_prob = torch.exp(total_prob0) / (torch.exp(total_prob0) + torch.exp(total_prob1)) #probability, that the human will chose trajectory0 over trajectory1
        max_prob = torch.max(total_prob0, total_prob1)
        predicted_prob = torch.exp(total_prob0 - max_prob) / (
                torch.exp(total_prob0 - max_prob) + torch.exp(total_prob1 - max_prob)
        )
        return predicted_prob


    def compute_predicted_probabilities(self, trajectories):
        predictions = [
            self.compute_predicted_probability(reward_model, trajectories)
            for reward_model in self.reward_networks
        ]
        return predictions

    def _compute_loss(self, reward_model, sample):
        """
        :param sample: A list of tuples where each tuple contains a pair of trajectories and their corresponding human feedback label.
                       The human feedback label is an indicator of which trajectory is preferred by the human.
        :return: The average entropy loss computed across all trajectory pairs in the sample. This represents the discrepancy
                 between the predicted probabilities and the human-provided feedback.
        """
        entropy_loss = 0.0
        for trajectory_pair, human_feedback_label in sample:
            predicted_prob = self.compute_predicted_probability(reward_model, trajectory_pair)
            # human feedback label to tensor conversion for processing
            label_1 = torch.tensor(human_feedback_label[0], dtype=torch.float)
            label_2 = 1 - label_1

            # calculate loss
            loss_1 = label_1 * torch.log(predicted_prob+1e-6)
            loss_2 = label_2 * torch.log(1 - predicted_prob + 1e-6)


            entropy_loss += -(loss_1 + loss_2)

        return entropy_loss #loss for whole batch


    def train_reward_models(self, pb, sample_size):
        model_losses = []
        val_losses = []

        for reward_model, optimizer in zip(self.reward_networks, self.optimizers):
            # Individual sampling
            sample = pb.sample(sample_size, replace=True)
            val_sample = pb.sample(int(sample_size / 2.718), replace=False)

            # Compute loss for this model
            model_loss = self._compute_loss(reward_model, sample)

            # Training for this model
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            model_losses.append(model_loss.item())

            with torch.no_grad():
                val_loss = self._compute_loss(reward_model, val_sample)
                val_losses.append(val_loss.item())

        # Average entropy loss over all models
        avg_entropy_loss = sum(model_losses) / len(model_losses)
        avg_overfit_loss = sum(val_losses) / len(val_losses)

        ratio = avg_overfit_loss / avg_entropy_loss
        if ratio < 1.1:
            self.l2 *= 1.1
            self._update_optimizers()
        elif ratio > 1.5:
            self.l2 *= 0.9
            self._update_optimizers()

        return avg_entropy_loss, ratio