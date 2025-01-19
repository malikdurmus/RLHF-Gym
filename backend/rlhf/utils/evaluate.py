import gymnasium as gym
import numpy as np
import torch
import gymnasium as gym

def evaluate_agent(eval_env_id, max_steps,  n_eval_episodes , actor_policy ,
        device, idx , capture_video, run_name, record_every_th_episode):
    env = gym.make(eval_env_id, render_mode="rgb_array", terminate_when_unhealthy=True,
                   exclude_current_positions_from_observation=False, capture_video=capture_video,record_every_th_episode=record_every_th_episode)

    env.reset()

    episode_rewards = []
    for episode in range(n_eval_episodes):
        obs = env.reset() #TODO:
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            actions, _, _ = actor_policy.get_action(torch.Tensor(obs).to(device))
            new_state, reward, done, info = env.step(actions)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

