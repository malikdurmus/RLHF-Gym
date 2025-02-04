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


    def train_reward_models(self, pb, batch_size):
        model_losses = []
        val_losses = []
        # Preference buffer has fewer entries than the sample batch
        if batch_size > len(pb):
            batch_size = len(pb)

        for reward_model, optimizer in zip(self.reward_networks, self.optimizers):
            # Individual sampling
            sample, val_sample = pb.sample_with_validation_sample(batch_size, replace=True)

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
        if ratio < 1.1:
            self.l2 *= 1.1
            self._update_optimizers()
        elif ratio > 1.5:
            self.l2 *= 0.9
            self._update_optimizers()

        return avg_entropy_loss, ratio

    def compute_predicted_probabilities(self, trajectories):
        predictions = [
            self._compute_predicted_probability_batch(reward_model, trajectories)
            for reward_model in self.reward_networks
        ]
        return predictions




    def _compute_loss_batch(self, reward_model, sample):
        """
        Compute the cumulative entropy loss for the given sample.
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

    def _compute_predicted_probability_batch(self, reward_model, trajectories):
        """
        Compute the predicted probability of human preference for trajectory0 over trajectory1.
        """
        trajectory0, trajectory1 = trajectories

        if trajectory0.states.size(0) != trajectory1.states.size(0):
            raise ValueError("Trajectory lengths do not match.")

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