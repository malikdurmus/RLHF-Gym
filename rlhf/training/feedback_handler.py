import uuid
from rlhf.rendering.gym_renderer import render_trajectory_gym


def handle_feedback(args, global_step, video_queue, stored_pairs, preference_buffer,
                    feedback_event, notify, trajectory_pairs, run_name):

    if not args.synthetic_feedback: # human feedback
        for query in range(len(trajectory_pairs)):
            trajectory_pair = trajectory_pairs[query]
            _render_and_queue_trajectories(args, query, global_step, video_queue, stored_pairs, trajectory_pair, run_name)

        if not video_queue.qsize() == len(trajectory_pairs):
            raise Exception("queue has more/less entries")
        elif video_queue.qsize() == len(trajectory_pairs):
            notify()

        print("Waiting for user feedback...")
        feedback_event.wait()  # Wait until feedback is populated
        feedback_event.clear()  # Reset the event
        stored_pairs.clear()
    else:
        for query in range(len(trajectory_pairs)):
            trajectory_pair = trajectory_pairs[query]
            _handle_synthetic_feedback(preference_buffer, trajectory_pair)


def _render_and_queue_trajectories(args, query, global_step, video_queue, stored_pairs, trajectory_pair, run_name):

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
    return traj.env_rewards.sum().item()
