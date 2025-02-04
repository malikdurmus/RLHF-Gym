import imageio
import numpy as np
import torch
from rlhf.environment.utils import initialize_eval_env


def evaluation(env_id, max_steps, n_eval_episodes, actor_policy, device , out_directory):
    #Record a replay video of the agent
    record_replay_video(env_id,max_steps, actor_policy, out_directory, device)
    #Evaluate the agent based on rewards
    evaluate_agent(env_id, max_steps, n_eval_episodes, actor_policy, device)


def evaluate_agent(eval_env_id, max_steps, n_eval_episodes, actor_policy, device):
    """
    Evaluates the agent's performance in a given environment over multiple episodes.

    This function runs the agent in an evaluation environment, collects rewards for
    each episode, and computes the mean and standard deviation of the rewards
    across all episodes.

    Args:
        eval_env_id (str): The ID of the environment to evaluate the agent on (e.g., "CartPole-v1").
        max_steps (int): Maximum number of steps to run the agent in each evaluation episode.
        n_eval_episodes (int): Number of evaluation episodes to run the agent.
        actor_policy (Actor): The trained actor policy (a model) used to generate actions for the agent.
        device (torch.device): The device (CPU or GPU) where the model and tensor computations will occur.

    Returns:
        None: This function prints the mean and standard deviation of the rewards from all evaluation episodes.

    Notes:
        - The function assumes that `initialize_eval_env` is defined elsewhere to create the environment.
        - The actor policy's `get_action` method should return the action as a tensor.
        - The environment should implement the `reset` and `step` methods compatible with Gym-style environments.
    """

    # Initialize the evaluation environment
    env = initialize_eval_env(eval_env_id)
    state, _ = env.reset()  # Reset environment and get initial state

    episode_rewards = []  # List to store total rewards for each episode

    # Evaluate agent over multiple episodes
    for episode in range(n_eval_episodes):
        state, _ = env.reset()  # Reset environment at the start of each episode
        total_rewards_ep = 0  # Track total reward for this episode

        # Evaluate agent within each episode, up to the max number of steps
        for step in range(max_steps):
            input_t = torch.Tensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
            input_t = input_t.to(device)  # Move tensor to the specified device (GPU/CPU)

            # Get the action from the actor policy
            action, _, _ = actor_policy.get_action(input_t)

            # Take the action in the environment, get new state and reward
            new_state, reward, done, terminated, info = env.step(action.detach().cpu().numpy().squeeze())

            # Accumulate the total reward for the current episode
            total_rewards_ep += reward

            # If episode is done or terminated, break out of the loop
            if done or terminated:
                break

            state = new_state  # Update state to the new state for the next step

        # Store the total reward for this episode
        episode_rewards.append(total_rewards_ep)

    # Calculate the mean and standard deviation of the rewards across all evaluation episodes
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    # Print the mean and standard deviation of the rewards
    print(mean_reward, std_reward)


def record_replay_video(env_id, max_steps, actor_policy, out_directory, device):
    """
    Generate a replay video of the agent's performance in the evaluation environment.

    This function records the agent's actions in the environment, renders the frames,
    and saves them as a video in the specified directory.

    Args:
        env_id (str): The ID of the environment in which the agent will perform (e.g., "CartPole-v1").
        max_steps (int): Maximum number of steps to run the agent in each replay video.
        actor_policy (Actor): The trained actor policy (a model) used to generate actions for the agent.
        out_directory (str): The directory where the output video will be saved.
        device (torch.device): The device (CPU or GPU) where the model and tensor computations will occur.

    Returns:
        None: This function saves a replay video of the agent's performance in the specified output directory.

    Notes:
        - The function assumes that `initialize_eval_env` is defined elsewhere to create the environment.
        - The actor policy's `get_action` method should return the action as a tensor.
        - The environment should implement the `reset` and `step` methods compatible with Gym-style environments.
        - The video is saved with a frame rate of 60 FPS.
    """

    env = initialize_eval_env(env_id)
    images = []  # List to store rendered frames from the environment
    terminated = False
    done = False
    state, _ = env.reset()

    # Capture the first frame
    img = env.render()
    images.append(img)
    steps  = 0

    # Perform the agent's actions and record the frames
    while (not done) and (not terminated) and steps <= max_steps:

        input_t = torch.Tensor(state).unsqueeze(0)
        input_t = input_t.to(device)

        # Get the action from the actor policy
        action, _, _ = actor_policy.get_action(input_t)

        # Take the action in the environment and observe the new state, reward, etc.
        new_state, reward, done, terminated, info = env.step(action.detach().cpu().numpy().squeeze())

        # Render the current state and store the image
        img = env.render()
        images.append(img)

        # Update state for the next step
        state = new_state
        steps = steps + 1

    # Save the recorded frames as a video
    with imageio.get_writer(out_directory + "/evaluation_video.mp4", fps=60) as writer:
        for img in images:
            writer.append_data(img)

    print("Saved evaluation video to: ", out_directory + "/evaluation_video.mp4")
