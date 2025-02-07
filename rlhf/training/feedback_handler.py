import uuid
from rlhf.rendering.gym_renderer import render_trajectory_gym


def handle_feedback(args, global_step, video_queue, stored_pairs, preference_buffer,
                    feedback_event, notify, trajectory_pairs, run_name):
    """
    Handles feedback collection for trajectory pairs, either from a human or synthetic source.

    If human feedback is enabled, it renders and queues trajectories for visualization and waits for user input.
    If synthetic feedback is enabled, it automatically generates feedback based on trajectory rewards.

    :param args: Configuration arguments controlling feedback behavior.
    :param global_step: The current global step in the training process.
    :param video_queue: A queue to store rendered trajectory videos for human feedback.
    :param stored_pairs: A list to store trajectory pairs and their metadata.
    :param preference_buffer: The buffer to store feedback (preferences) for trajectory pairs.
    :param feedback_event: An event to signal when feedback has been received.
    :param notify: A function to notify when rendering is complete.
    :param trajectory_pairs: A list of trajectory pairs to collect feedback for.
    :param run_name: The name of the current run, used for organizing rendered videos.

    :raises Exception: If the video queue size does not match the number of trajectory pairs.
    """

    if not args.synthetic_feedback: # human feedback
        for query in range(len(trajectory_pairs)):
            trajectory_pair = trajectory_pairs[query]
            _render_and_queue_trajectories(args, query, global_step, video_queue, stored_pairs, trajectory_pair, run_name)

        if not video_queue.qsize() == len(trajectory_pairs):
            raise Exception("queue has more/less entries")
        elif video_queue.qsize() == len(trajectory_pairs):
            notify()

        print("Waiting for user feedback...")
        print("(http://localhost:5000)")
        feedback_event.wait()  # Wait until feedback is populated
        feedback_event.clear()  # Reset the event
        stored_pairs.clear()
    else:
        for query in range(len(trajectory_pairs)):
            trajectory_pair = trajectory_pairs[query]
            _handle_synthetic_feedback(preference_buffer, trajectory_pair)


def _render_and_queue_trajectories(args, query, global_step, video_queue, stored_pairs, trajectory_pair, run_name):
    """
    Renders and queues trajectory pairs for visualization and feedback collection.
    :param args: Configuration arguments controlling rendering behavior.
    :param query: The index of the current trajectory pair being processed.
    :param global_step: The current global step in the training process.
    :param video_queue: A queue to store rendered trajectory videos.
    :param stored_pairs: A list to store trajectory pairs and their metadata.
    :param trajectory_pair: The trajectory pair to render and queue.
    :param run_name: The name of the current run, used for organizing rendered videos.
    """
    trajectory1, trajectory2 = trajectory_pair

    # Notify that rendering has started
    print(f"Requested rendering for query {query} at step {global_step}")
    video1_path = render_trajectory_gym(
        args.env_id, trajectory1, global_step, "trajectory1", query, run_name)
    video2_path = render_trajectory_gym(
        args.env_id, trajectory2, global_step, "trajectory2", query, run_name)

    pair_id = str(uuid.uuid4())  # UUID generation
    video_queue.put((pair_id, trajectory1, trajectory2, video1_path, video2_path))
    stored_pairs.append({
        'id': pair_id,
        'trajectory1': trajectory1,
        'trajectory2': trajectory2
    })

def _handle_synthetic_feedback(preference_buffer, trajectory_pair):
    """
    Generates synthetic feedback for a trajectory pair based on their cumulative rewards.

    The feedback is determined by comparing the rewards of the two trajectories. If the difference
    is within a dynamic threshold, the preference is neutral. Otherwise, the trajectory with the
    higher reward is preferred.

    :param preference_buffer: The buffer to store feedback (preferences) for trajectory pairs.
    :param trajectory_pair: The trajectory pair to generate feedback for.
    """
    trajectory1, trajectory2 = trajectory_pair

    rewards_1 = _sum_rewards(trajectory1)
    rewards_2 = _sum_rewards(trajectory2)

    # calculate dynamic threshold
    larger_reward = max(abs(rewards_1), abs(rewards_2))
    threshold = 0.1 * larger_reward

    # if rewards only differ by 10 percent, preference is neutral
    if abs(rewards_1 - rewards_2) <= threshold:
        preference = 0.5
    elif _sum_rewards(trajectory1) > _sum_rewards(trajectory2):
        preference = 1
    else:
        preference = 0
    preference_buffer.add((trajectory1, trajectory2), preference)

def _sum_rewards(traj):
    """
    Computes the sum of rewards for a given trajectory.
    :param traj: The trajectory for which to compute the sum of rewards.

    :return: The sum of rewards for the trajectory.
    """
    return traj.env_rewards.sum().item()
