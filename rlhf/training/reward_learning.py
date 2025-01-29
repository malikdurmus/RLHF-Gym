import torch
import torch.optim as optim

class PreferencePredictor:
    def __init__(self, reward_networks: list, reward_model_lr, device, l2):
        """Constructor for the PreferencePredictor
        :param reward_networks: a list of reward networks
        :param reward_model_lr: reward network learn rate
        :param device: "cuda" or "cpu"
        :param l2: l2 parameter for overfitting handling
        """
        self.reward_networks = reward_networks
        self.device = device
        self.l2 = l2
        self.reward_model_lr = reward_model_lr
        self.optimizers = self._initialize_optimizers()

    def _initialize_optimizers(self):
        """Initialize optimizers for reward networks
        :param None
        :return: A list with optimizers for each reward_network we have
        """
        return [
            optim.Adam(
                reward_network.parameters(),
                lr=self.reward_model_lr,
                weight_decay=self.l2
            )
            for reward_network in self.reward_networks
        ]

    def _update_optimizers(self):
        """Update optimizers (Is used when overfitting occurs)
        :param None
        :return: None
        """
        self.optimizers = self._initialize_optimizers()


    def train_reward_models(self, pb, batch_size):
        """Train our reward networks with calculated average of entropy losses over networks
        :param pb: PreferenceBuffer which we sample from
        :param batch_size: number of preferences
        :return: avg_entropy_loss: the average entropy loss of our reward networks
        :return: ratio: the ratio of our entropy loss from our validation sample and model entropy loss
        """
        model_losses = []
        val_losses = []
        # Preference buffer has fewer entries than the sample batch
        if batch_size > len(pb):
            batch_size = len(pb)

        for reward_model, optimizer in zip(self.reward_networks, self.optimizers):
            # Individual sampling # TODO The two samples shouldn't overlap
            sample = pb.sample(batch_size, replace=True)
            val_sample = pb.sample(int(batch_size / 2.718), replace=False) # Validation sample

            # Compute loss for this model
            model_loss = self._compute_loss_batch(reward_model, sample)

            # Training for this model
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            model_losses.append(model_loss.item())

            # Use validation sample with trained reward model (to test for overfitting)
            with torch.no_grad():
                val_loss = self._compute_loss_batch(reward_model, val_sample)
                val_losses.append(val_loss.item())

        # Average losses over all models
        avg_entropy_loss = sum(model_losses) / len(model_losses)
        avg_overfit_loss = sum(val_losses) / len(val_losses)

        # Calculate ratio
        ratio = avg_overfit_loss / avg_entropy_loss

        # Adjust coefficient to keep validation loss between 1.1 and 1.5
        # TODO need to adjust how much the coefficient is changed (maybe use some function including ratio and target)
        if ratio < 1.1:
            self.l2 *= 1.1
            self._update_optimizers()
        elif ratio > 1.5:
            self.l2 *= 0.9
            self._update_optimizers()

        return avg_entropy_loss, ratio

    def compute_predicted_probabilities(self, trajectories):
        """Compute the predicted probability of human preference for trajectory0 over trajectory1. (Over all networks)
        :param trajectories: a trajectory tuple (traj1,traj2)
        :return: predictions: a list of predicted probabilities that trajectory0 is chosen over trajectory1
                 (calculated by each of our networks) for a trajectory tuple
        """
        predictions = [
            self._compute_predicted_probability_batch(reward_model, trajectories)
            for reward_model in self.reward_networks
        ]
        return predictions




    def _compute_loss_batch(self,reward_model, sample):
        """Compute the cumulative entropy loss for the given sample
        :param reward_model: the reward model for which we want to calculate an entropy loss
        :param sample: a sample(with size batch_size) taken out of the Preference buffer Form: [((traj1,traj2),pref)...]
        :return: entropy_loss entropy loss we calculate for that sample
                 (entropy loss=difference between label and predicted prob)
        """
        entropy_loss = torch.tensor(0.0, requires_grad=True)

        for trajectory_pair, human_feedback_label in sample:
            predicted_prob = self._compute_predicted_probability_batch(reward_model,trajectory_pair)

            # label to tensor
            label = torch.tensor(human_feedback_label, dtype=torch.float, device=self.device)

            # clamp to to avoid log(0)
            predicted_prob = torch.clamp(predicted_prob, min=1e-7, max=1 - 1e-7)

            loss = -label * torch.log(predicted_prob) - (1 - label) * torch.log(1 - predicted_prob)
            entropy_loss = entropy_loss + loss

        return entropy_loss

    def _compute_predicted_probability_batch(self,reward_model, trajectories):
        """Compute the predicted probability of human preference for trajectory0 over trajectory1.
        :param reward_model: reward network which calculates its prediction
        :param trajectories: a trajectory pair (the traj pair of our sample)
        :return: the predicted probability(of our network) that traj0 is favored over traj1
        """
        trajectory0, trajectory1 = trajectories

        if trajectory0.states.size(0) != trajectory1.states.size(0):
            raise ValueError("Trajectory lengths do not match.")

        trajectory0.to(self.device)
        trajectory1.to(self.device)

        rewards0 = reward_model(trajectory0.actions, trajectory0.states)
        rewards1 = reward_model(trajectory1.actions, trajectory1.states)

        total_reward0 = rewards0.sum()
        total_reward1 = rewards1.sum()

        # softmax
        max_reward = torch.max(total_reward0, total_reward1)
        predicted_prob = torch.exp(total_reward0 - max_reward) / (
                torch.exp(total_reward0 - max_reward) + torch.exp(total_reward1 - max_reward)
        )

        return predicted_prob





# Deprecated Methods, will remove those
# for each data in sample we will use the predict function. after that we will use the preference to calculate entropy loss
    def _deprecated_predicted_probability(self,reward_model, trajectories):
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
            prob_for_action = reward_model(action=action,
                                                  observation=state) # estimated probability, that the human will prefer action 0
            total_prob0 += prob_for_action

        for state, action in zip(states1, actions1):
            action = action.to(self.device)
            state = state.to(self.device)
            prob_for_action = reward_model(action=action,
                                                  observation=state)  # estimated probability, that the human will prefer action 1
            total_prob1 += prob_for_action # Tensor of shape {Tensor : {1,1}} , tested

        max_prob = torch.max(total_prob0, total_prob1)
        predicted_prob = torch.exp(total_prob0 - max_prob) / (
                torch.exp(total_prob0 - max_prob) + torch.exp(total_prob1 - max_prob)
        )
        return predicted_prob

    def _deprecated_compute_loss(self, reward_model,sample):
        """
        :param sample: A list of lists where each inner list contains:
               - A tuple (trajectory1, trajectory2) (both trajectories)
               - An integer representing human feedback (0 or 1)
        :return: The cumulative entropy loss computed across all trajectory pairs in the sample.
        """
        entropy_loss = torch.tensor(0.0, requires_grad=True)
        for trajectory_pair, human_feedback_label in sample:
            predicted_prob = self._deprecated_predicted_probability(reward_model,trajectory_pair)
            # human feedback label to tensor conversion for processing
            label_1 = torch.tensor(human_feedback_label, dtype=torch.float)
            label_1 = label_1.to(self.device)

            predicted_prob = torch.clamp(predicted_prob, min=1e-7, max=1 - 1e-7)

            # calculate loss
            loss_1 = label_1 * torch.log(predicted_prob)
            loss_2 = (1 - label_1) * torch.log(1 - predicted_prob)

            entropy_loss = entropy_loss + -(loss_1 + loss_2 )
        return entropy_loss #loss for whole batch
