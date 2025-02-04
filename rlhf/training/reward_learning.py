import torch
import torch.optim as optim

class PreferencePredictor:
    """
    A class that predicts human preferences between trajectory pairs using an ensemble of reward models.

    The class handles the training of multiple reward models, both for supervised and semi-supervised tasks,
    using preference data. It also includes logic for adjusting model parameters based on validation losses.

    Attributes:
        reward_networks (list): A list of reward models (neural networks) that predict human preferences.
        device: The device on which the models and data are loaded (e.g., "cuda" or "cpu").
        l2 (float): L2 regularization coefficient for the optimizer.
        reward_model_lr (float): Learning rate for the reward models.
        optimizers (list): A list of Adam optimizers, one for each reward model in the ensemble.
    """

    def __init__(self, reward_networks: list, reward_model_lr, device, l2):
        """
        Initializes the PreferencePredictor with the specified parameters.

        Args:
            reward_networks (list): List of reward models (neural networks) used for predicting preferences.
            reward_model_lr (float): Learning rate for the reward models.
            device: The device (e.g., "cuda" or "cpu") where models and data are loaded.
            l2 (float): The L2 regularization coefficient used in the optimizer.
        """
        self.reward_networks = reward_networks
        self.device = device
        self.l2 = l2
        self.reward_model_lr = reward_model_lr
        self.optimizers = self._initialize_optimizers()

    def _initialize_optimizers(self):
        """
        Initializes Adam optimizers for each reward network with the specified learning rate and L2 regularization.

        Returns:
            list: A list of Adam optimizers, one for each reward model.
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
        """
        Re-initializes the optimizers to apply the updated L2 regularization value.
        """
        self.optimizers = self._initialize_optimizers()

    def train_reward_models(self, pb, batch_size):
        """
        Trains the reward models using the provided preference buffer and batch size.

        For each reward model, it computes the loss, performs backpropagation,
        and updates the model parameters using an optimizer. It also validates the models
        using a validation subset of the preference buffer.

        Args:
            pb: The preference buffer containing human-labeled trajectory pairs.
            batch_size (int): The batch size used for training.

        Returns:
            tuple: A tuple containing:
                - avg_entropy_loss (float): The average entropy loss over all models.
                - ratio (float): The ratio of the validation loss to the training loss.
        """
        model_losses = []
        val_losses = []

        # Preference buffer has fewer entries than the sample batch, adjust batch size if necessary
        if batch_size > len(pb):
            batch_size = len(pb)

        for reward_model, optimizer in zip(self.reward_networks, self.optimizers):
            # Individual sampling for each model
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

        # Calculate ratio of validation loss to training loss
        ratio = avg_overfit_loss / avg_entropy_loss

        # Adjust L2 regularization based on the ratio to avoid overfitting: Keep validation loss between 1.1 and 1.5
        if ratio < 1.1:
            self.l2 *= 1.1
            self._update_optimizers()
        elif ratio > 1.5:
            self.l2 *= 0.9
            self._update_optimizers()

        return avg_entropy_loss, ratio

    def train_reward_models_surf(self, augmented_preference_buffer, ssl_preference_buffer, batch_size, loss_weight_ssl):
        """
        Trains the reward models using both augmented human-labeled data and pseudo-labeled data from SSL.

        This function combines the human-labeled data in the augmented preference buffer and the pseudo-labeled
        data in the SSL buffer, applying the given weight to the pseudo-labeled loss.

        Args:
            augmented_preference_buffer: The preference buffer with human-labeled trajectory pairs, possibly augmented.
            ssl_preference_buffer: The preference buffer containing pseudo-labeled trajectory pairs.
            batch_size (int): The batch size used for training.
            loss_weight_ssl (float): The weight applied to the loss from pseudo-labeled data.

        Returns:
            tuple: A tuple containing:
                - entropy_loss (float): The average entropy loss over all models.
                - ratio (float): The ratio of the validation loss to the training loss.
        """
        model_losses = []
        val_losses = []

        # If batch_size is not provided, set it to a default value
        if batch_size is None:
            batch_size = 32  # Default batch size

        loss_w_ssl = loss_weight_ssl

        # Ensure batch size does not exceed the size of the augmented preference buffer
        if batch_size > len(augmented_preference_buffer):
            batch_size = len(augmented_preference_buffer)

        for reward_model, optimizer in zip(self.reward_networks, self.optimizers):
            # Sample human-labeled and validation samples from the augmented preference buffer
            human_labeled_sample, human_labeled_val_sample = (augmented_preference_buffer.
                                                              sample_with_validation_sample(batch_size, replace=True))


            # Use the SSL buffer directly for pseudo-labeled data
            pseudo_labeled_sample = ssl_preference_buffer.get_buffer()

            # Compute loss for human-labeled and pseudo-labeled data
            human_label_loss = self._compute_loss_batch(reward_model, human_labeled_sample)
            pseudo_label_loss = self._compute_loss_batch(reward_model, pseudo_labeled_sample)

            # Total loss is the sum of human-labeled loss and weighted pseudo-labeled loss
            total_loss = human_label_loss + pseudo_label_loss * loss_w_ssl

            # Training for this model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            model_losses.append(total_loss.item())

            # Validation loss to check for overfitting
            with torch.no_grad():
                val_loss = self._compute_loss_batch(reward_model, human_labeled_val_sample)
                val_losses.append(val_loss.item())

        # Return average loss and ratio of validation to training loss
        entropy_loss = sum(model_losses) / len(model_losses)
        ratio = sum(val_losses) / len(val_losses)
        return entropy_loss, ratio

    def compute_predicted_probabilities(self, trajectories):
        """
        Computes predicted probabilities for human preference between two trajectories from each reward model.

        Args:
            trajectories (tuple): A tuple containing two trajectory objects (trajectory0, trajectory1).

        Returns:
            list: A list of predicted probabilities from each reward model in the ensemble.
        """
        predictions = [
            self.compute_predicted_probability_batch(reward_model, trajectories)
            for reward_model in self.reward_networks
        ]
        return predictions

    def _compute_loss_batch(self, reward_model, sample):
        """
        Computes the cumulative entropy loss for the given sample from the preference buffer.

        Args:
            reward_model: The reward model used to compute preferences.
            sample: A batch of trajectory pairs and their corresponding human feedback labels.

        Returns:
            torch.Tensor: The cumulative entropy loss for the batch.
        """
        entropy_loss = torch.tensor(0.0, requires_grad=True)

        for trajectory_pair, human_feedback_label in sample:
            predicted_prob = self.compute_predicted_probability_batch(reward_model, trajectory_pair)

            # Convert label to tensor
            label = torch.tensor(human_feedback_label, dtype=torch.float, device=self.device)

            # Clamp predicted probability to avoid log(0)
            predicted_prob = torch.clamp(predicted_prob, min=1e-7, max=1 - 1e-7)

            # Compute binary cross-entropy loss
            loss = -label * torch.log(predicted_prob) - (1 - label) * torch.log(1 - predicted_prob)
            entropy_loss = entropy_loss + loss

        return entropy_loss

    def compute_predicted_probability_batch(self, reward_model, trajectories):
        """
        Computes the predicted probability of human preference for trajectory0 over trajectory1.

        Args:
            reward_model: The reward model used to compute preferences.
            trajectories (tuple): A tuple containing two trajectory objects (trajectory0, trajectory1).

        Returns:
            torch.Tensor: The predicted probability of human preference for trajectory0 over trajectory1.
        """
        trajectory0, trajectory1 = trajectories

        # Ensure trajectories have the same length
        if trajectory0.states.size(0) != trajectory1.states.size(0):
            raise ValueError("Trajectory lengths do not match.")

        trajectory0.to(self.device)
        trajectory1.to(self.device)

        rewards0 = reward_model(trajectory0.actions, trajectory0.states)
        rewards1 = reward_model(trajectory1.actions, trajectory1.states)

        total_reward0 = rewards0.sum()
        total_reward1 = rewards1.sum()

        # Apply softmax to compute probability
        max_reward = torch.max(total_reward0, total_reward1)

        predicted_prob = torch.exp(total_reward0 - max_reward) / ( # estimated probability, that the human will prefer action 0 over 1
            torch.exp(total_reward0 - max_reward) + torch.exp(total_reward1 - max_reward)
        )

        return predicted_prob
