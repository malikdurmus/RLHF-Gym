import torch
import numpy as np

from rlhf.buffers.preference_buffer import PreferenceBuffer, tda


def semi_supervised_labeling(preference_predictor,sampler,args):
    """
    Perform semi-supervised labeling using the ensemble of reward models.

    :param sampler:
    :param preference_predictor: List of reward models in the ensemble.
    :variable confidence_threshold: Confidence threshold for filtering pseudo-labels.
    :return: A new preference buffer containing both human-labeled and pseudo-labeled data.
    """
    # Initialize a new preference buffer for SSL
    ssl_preference_buffer = PreferenceBuffer(args.pref_batch_size)

    # 100 times more trajectory and preference pairs, since most of them will be filtered out via confidence_threshold
    labeled_batch_size = args.ensemble_query_size * 100
    confidence_threshold = args.confidence_threshold

    # Generate pseudo-labels for unlabeled data in the replay buffer
    unlabeled_trajectories_list = sampler.uniform_trajectory_pair_batch(args.traj_length,
                                                              args.feedback_frequency, labeled_batch_size,args.synthetic_feedback)

    for trajectory_pair in unlabeled_trajectories_list:
        # Apply Temporal Data Augmentation (TDA)
        try:
            cropped_trajectory_pair = tda(
                trajectory_pair, min_length=args.min_crop_length, max_length=args.max_crop_length
            )
        except ValueError as e:
            print(f"Skipping trajectory pair in human_feedback_pb due to error in TDA: {e}")
            continue

        # Compute the preference probability using the ensemble of reward models
        with torch.no_grad():
            # Get predictions from each reward model in the ensemble
            predictions = preference_predictor.compute_predicted_probabilities(cropped_trajectory_pair)

            predictions_stacked = torch.stack(predictions)
            # Take the mean
            mean_preference_prob = predictions_stacked.mean().item()

        # Generate pseudo-label
        if mean_preference_prob > 0.5:
            pseudo_label = 0  # segment_0 is preferred
            confidence = mean_preference_prob  # Confidence is the predicted probability
        else:
            pseudo_label = 1  # segment_1 is preferred
            confidence = 1 - mean_preference_prob  # Confidence is 1 - predicted probability

        # Filter and add pseudo-labels based on confidence threshold
        if confidence > confidence_threshold:
            ssl_preference_buffer.add(cropped_trajectory_pair , pseudo_label)

    return ssl_preference_buffer
