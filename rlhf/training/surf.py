import torch
from rlhf.buffers.preference_buffer import PreferenceBuffer, tda


def semi_supervised_labeling(preference_predictor, sampler, args, tda_active):
    """
    Perform semi-supervised labeling using an ensemble of reward models.

    This function generates pseudo-labels for unlabeled trajectory pairs based on the predictions
    of an ensemble of reward models. It filters the generated pseudo-labels based on a confidence
    threshold and returns a new preference buffer containing both human-labeled and pseudo-labeled data.

    Args:
        preference_predictor (PreferencePredictor): PreferencePredictor object. Contains reward models in the ensemble used to predict preferences and the necessary methods.
        sampler: The sampler used to generate trajectory pairs.
        args: Configuration arguments containing parameters such as batch size, confidence threshold, etc.
        tda_active (bool): Flag indicating whether temporal data augmentation (TDA) should be applied.

    Returns:
        PreferenceBuffer: A preference buffer containing both human-labeled and pseudo-labeled data.
    """
    # Initialize a new preference buffer for SSL // ssl buffer is recreated everytime to ensure the
    # reward_model update happens with the most reliable (last) labels.
    ssl_preference_buffer = PreferenceBuffer(args.preference_batch_size * 5)

    # Set labeled batch size and confidence threshold
    # 4 times more trajectory and preference pairs, since most of them will be filtered out via confidence_threshold
    # --this is fixed in SURF implementation
    labeled_batch_size = args.preference_batch_size * 4
    confidence_threshold = args.confidence_threshold

    # Generate unlabeled trajectory pairs for pseudo-labeling
    unlabeled_trajectories_list = sampler.uniform_trajectory_pair_batch(
        labeled_batch_size, args.trajectory_length, args.feedback_frequency, args.synthetic_feedback
    )

    for trajectory_pair in unlabeled_trajectories_list:
        # Apply Temporal Data Augmentation (TDA) if active
        if tda_active:
            try:
                trajectory_pair = tda(
                    trajectory_pair, crop_size=args.crop
                )
            except ValueError as e:
                print(f"Skipping trajectory pair in human_feedback_pb due to error in TDA: {e}")
                continue

        # Compute preference probabilities using the ensemble of reward models
        with torch.no_grad():
            predictions = preference_predictor.compute_predicted_probabilities(trajectory_pair)
            predictions_stacked = torch.stack(predictions)
            mean_preference_prob = predictions_stacked.mean().item()

        # Generate pseudo-label and its confidence based on mean preference probability
        if mean_preference_prob > 0.5:
            pseudo_label = 1  # segment_0 is preferred  TODO: fix the label mismatch in the project (works, but doesnt make sense to label 1 when feedback is 0)
            confidence = mean_preference_prob  # Confidence is the predicted probability
        else:
            pseudo_label = 0  # segment_1 is preferred
            confidence = 1 - mean_preference_prob  # Confidence is 1 - predicted probability

        # Filter and add pseudo-labels to preference buffer based on confidence threshold
        if confidence > confidence_threshold:
            ssl_preference_buffer.add(trajectory_pair, pseudo_label)

    return ssl_preference_buffer

def surf(preference_optimizer, sampler, args, preference_buffer, recent_data_size):
    """
    Implements the SURF (Semi-Supervised Reward Learning with Feedback) algorithm.

    This function handles the semi-supervised learning pipeline by incorporating the use of temporal data
    augmentation (TDA) and semi-supervised labeling (SSL) to augment the preference buffer and train reward models.

    Args:
        preference_optimizer: The optimizer used for training the reward models.
        sampler: The sampler used for generating trajectory pairs.
        args: Configuration arguments that control the behavior of the algorithm.
        preference_buffer: The buffer containing human-labeled preference data.
        recent_data_size: The number of elements each feedback query

    Returns:
        entropy_loss (float): The entropy loss incurred during the training.
        ratio (float): The ratio of positive to negative preferences used in training.

    Raises:
        Exception: If neither SSL nor TDA is enabled when SURF is activated.
    """
    if args.ssl and args.tda_active:  # Semi-supervised labeling with TDA (Trajectory Data Augmentation)
        ssl_preference_buffer = semi_supervised_labeling(preference_optimizer, sampler, args, tda_active=True)

        augmented_pb = preference_buffer.copy(
            apply_tda=True, crop_size= args.crop
        )

        entropy_loss, validation_loss, ratio, l2 = preference_optimizer.train_reward_models_surf(
            augmented_pb, ssl_preference_buffer, args.preference_batch_size,
            recent_data_size, loss_weight_ssl=args.loss_weight_ssl
        )
    elif args.ssl:  # Semi-supervised labeling without TDA
        ssl_preference_buffer = semi_supervised_labeling(preference_optimizer, sampler, args, tda_active=False)

        unchanged_preference_buffer = preference_buffer.copy(apply_tda=False)

        entropy_loss, validation_loss, ratio, l2 = preference_optimizer.train_reward_models_surf(
            unchanged_preference_buffer, ssl_preference_buffer, args.preference_batch_size, recent_data_size,
            loss_weight_ssl=args.loss_weight_ssl
        )
    elif args.tda_active:  # Only TDA (no semi-supervised labeling)
        augmented_pb = preference_buffer.copy(
            apply_tda=True, crop_size = args.crop
        )
        entropy_loss, validation_loss, ratio, l2 = preference_optimizer.train_reward_models(augmented_pb,
                                                                                            args.preference_batch_size,
                                                                                            recent_data_size)
    else:
        raise Exception("If SURF is True, either tda_active or ssl must be True")

    return entropy_loss, validation_loss, ratio, l2
