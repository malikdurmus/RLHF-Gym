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
        :param reward_networks: List of reward models (neural networks) used for predicting preferences.
        :param reward_model_lr: Learning rate for the reward models.
        :param device: The device (e.g., "cuda" or "cpu") where models and data are loaded.
        :param l2: The L2 regularization coefficient used in the optimizer.
        """
        self.reward_networks = reward_networks
        self.device = device
        self.l2 = l2
        self.reward_model_lr = reward_model_lr
        self.optimizers = self._initialize_optimizers()

    def _initialize_optimizers(self):
        """
        Initializes Adam optimizers for each reward network with the specified learning rate and L2 regularization.

        :return: A list of Adam optimizers, one for each reward model.
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

    def _adjust_l2(self, ratio):
        """
        Adjusts the L2 regularization coefficient based on the ratio of validation loss to training loss.
        :param ratio: The ratio of validation loss to training loss.
        """
        if ratio < 1.1:
            self.l2 *= 0.95  # Decrease L2 to allow more fitting
            self._update_optimizers()
        # If ratio is too high (overfitting), increase L2 to add regularization
        elif ratio > 1.5:
            self.l2 *= 1.05  # Increase L2 to reduce overfitting
            self._update_optimizers()

    def train_reward_models(self, pb, batch_size, recent_data_size):
        """
        Trains the reward models using the provided preference buffer and batch size.

        For each reward model, it computes the loss, performs backpropagation,
        and updates the model parameters using an optimizer. It also validates the models
        using a validation subset of the preference buffer.

        :param pb: The preference buffer containing human-labeled trajectory pairs.
        :param batch_size: The batch size used for training.
        :param recent_data_size: The number of elements each feedback query

        :return: A tuple containing:
        - avg_post_update_loss (float): The average training loss after update.
        - avg_val_loss (float): The average validation loss over all models.
        - ratio (float): The ratio of the validation loss to the training loss.
        - l2 (float): The updated L2 regularization coefficient.
        """
        val_losses = []
        post_update_losses = []


        # Preference buffer has fewer entries than the sample batch, adjust batch size if necessary
        if batch_size > len(pb):
            batch_size = len(pb)

        for reward_model, optimizer in zip(self.reward_networks, self.optimizers):
            # Individual sampling for each model
            sample, val_sample = pb.sample_with_validation_sample(batch_size, recent_data_size, replace=True)

            # Compute loss for this model
            model_loss = self._compute_loss_batch(reward_model, sample)

            # Training for this model
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            # Use validation sample with trained reward model
            # Check for Overfitting, Underfitting, Generalization
            with torch.no_grad():
                # Recompute the training loss on the same human-labeled sample (post-update)
                post_update_loss = self._compute_loss_batch(reward_model, sample)
                post_update_loss = post_update_loss / len(sample)  # Normalize
                post_update_losses.append(post_update_loss.item())

                # Compute validation loss
                val_loss = self._compute_loss_batch(reward_model, val_sample)
                val_loss = val_loss / len(val_sample) # Normalize
                val_losses.append(val_loss.item())

        # Return average loss and ratio of validation to training loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_post_update_loss = sum(post_update_losses) / len(post_update_losses)

        ratio = avg_val_loss / avg_post_update_loss

        # Adjust L2 regularization based on the ratio: Keep validation loss between 1.1 and 1.5
        self._adjust_l2(ratio)

        return avg_post_update_loss, avg_val_loss, ratio, self.l2

    def train_reward_models_surf(self, augmented_preference_buffer, ssl_preference_buffer,
                                 batch_size, recent_data_size, loss_weight_ssl):
        """
        Trains the reward models using both augmented human-labeled data and pseudo-labeled data from SSL.

        This function combines the human-labeled data in the augmented preference buffer and the pseudo-labeled
        data in the SSL buffer, applying the given weight to the pseudo-labeled loss.

        :param augmented_preference_buffer: The preference buffer with human-labeled trajectory pairs, possibly augmented.
        :param ssl_preference_buffer: The preference buffer containing pseudo-labeled trajectory pairs.
        :param batch_size: The batch size used for training.
        :param recent_data_size: The number of elements each feedback query
        :param loss_weight_ssl: The weight applied to the loss from pseudo-labeled data.

        :return: A tuple containing:
            - avg_post_update_loss (float): The average training loss after update.
            - avg_val_loss (float): The average validation loss.
            - ratio (float): The ratio of the validation loss to the training loss.
            - l2 (float): The updated L2 regularization coefficient.
        """
        val_losses = []
        post_update_losses = []

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
                                                              sample_with_validation_sample(batch_size, recent_data_size,
                                                                                            replace=True))


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

            # Check for Overfitting, Underfitting, Generalization
            with torch.no_grad():
                # Recompute the training loss on the same human-labeled sample (post-update)
                post_human_loss = self._compute_loss_batch(reward_model, human_labeled_sample) / len(human_labeled_sample)  # Normalize
                post_ssl_loss = self._compute_loss_batch(reward_model, pseudo_labeled_sample) / len(pseudo_labeled_sample)  # Normalize
                post_update_loss = post_human_loss + post_ssl_loss * loss_w_ssl
                post_update_losses.append(post_update_loss.item())

                # Compute validation loss
                val_loss = self._compute_loss_batch(reward_model, human_labeled_val_sample)
                val_loss = val_loss / len(human_labeled_val_sample) # Normalize
                val_losses.append(val_loss.item())

        # Return average loss and ratio of validation to training loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_post_update_loss = sum(post_update_losses) / len(post_update_losses)

        ratio = avg_val_loss / avg_post_update_loss

        # Adjust L2 regularization based on the ratio: Keep validation loss between 1.1 and 1.5
        self._adjust_l2(ratio)

        return avg_post_update_loss, avg_val_loss, ratio, self.l2

    def compute_predicted_probabilities(self, trajectories):
        """
        Computes predicted probabilities for human preference between two trajectories from each reward model.
        :param trajectories: A tuple containing two trajectory objects (trajectory0, trajectory1).

        :return: A list of predicted probabilities from each reward model
        """
        predictions = [
            self.compute_predicted_probability_batch(reward_model, trajectories)
            for reward_model in self.reward_networks
        ]
        return predictions

    def _compute_loss_batch(self, reward_model, sample):
        """
        Computes the cumulative entropy loss for the given sample from the preference buffer.
        :param reward_model: The reward model used to compute preferences.
        :param sample: A batch of trajectory pairs and their corresponding human feedback labels.

        :return: The cumulative entropy loss for the batch.
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
        :param reward_model: The reward model used to compute preferences.
        :param trajectories: A tuple containing two trajectory objects (trajectory0, trajectory1).

        :return: The predicted probability of human preference for trajectory0 over trajectory1.
        :raises ValueError: If the lengths of the two trajectories do not match.
        """
        trajectory0, trajectory1 = trajectories

        # Ensure trajectories have the same length
        if trajectory0.states.size(0) != trajectory1.states.size(0):
            raise ValueError("Trajectory lengths do not match.")

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