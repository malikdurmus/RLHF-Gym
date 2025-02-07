import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from rlhf.training.feedback_handler import handle_feedback
from rlhf.training.surf import surf

def train(envs, rb, actor, reward_networks, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer,
          preference_optimizer, args, writer, device, sampler,
          human_labeled_preference_buffer, video_queue, stored_pairs, feedback_event, int_rew_calc, notify, preference_mutex, run_name):
    """
    PEBBLE Algorithm logic
    """

    # [Optional] automatic adjustment of the entropy coefficient
    if args.automatic_entropy_coefficient_tuning:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_network_lr)
    else:
        alpha = args.entropy_regularization_coefficient

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    episodic_model_rewards = 0

    # calculate timesteps in which feedback is requested
    remaining_timesteps = args.total_timesteps - args.reward_learning_starts
    # calculate how many feedbacks there will be
    feedback_iterations = (remaining_timesteps + args.feedback_frequency - 1) // args.feedback_frequency
    # calculate how many queries we present each iteration
    query_size = max(1, args.total_queries // feedback_iterations)
    # calculate how many of the queries are ensemble-based
    num_ensemble = int(query_size * (args.ensemble_ratio / 100))
    # calculate how many of the queries are uniform-based
    num_uniform = query_size - num_ensemble


    ### PEBBLE ALGO ###
    for global_step in range(args.total_timesteps):
        ### REWARD LEARNING ###
        if global_step >= args.reward_learning_starts and global_step != 0:
            ### FEEDBACK QUERY ###
            if global_step % args.feedback_frequency == 0 or global_step == args.reward_learning_starts:
                trajectory_pairs = []
                # uniform-sampling
                if num_uniform > 0:
                    trajectory_pairs.extend(sampler.uniform_trajectory_pair_batch(num_uniform, args.trajectory_length,
                                                                             args.feedback_frequency,
                                                                             args.synthetic_feedback)
                                            )
                # ensemble-sampling
                if num_ensemble > 0:
                    trajectory_pairs.extend(sampler.ensemble_sampling(num_ensemble, args.trajectory_length,
                                                                 args.feedback_frequency, args.synthetic_feedback,
                                                                 preference_optimizer)
                                            )

                # handle feedback
                handle_feedback(args, global_step, video_queue, stored_pairs, human_labeled_preference_buffer,
                                feedback_event, notify, trajectory_pairs, run_name)

                ### REWARD MODEL TRAINING ###
                with preference_mutex:
                    if args.surf:
                        entropy_loss, validation_loss, ratio, l2 = surf(preference_optimizer, sampler, args, human_labeled_preference_buffer)
                    else:
                        entropy_loss, validation_loss, ratio, l2 = preference_optimizer.train_reward_models(human_labeled_preference_buffer,
                                                                                       args.preference_batch_size)

                writer.add_scalar("losses/entropy_loss", entropy_loss, global_step)
                writer.add_scalar("losses/validation_loss", validation_loss, global_step)
                writer.add_scalar("losses/validation_ratio", ratio, global_step)
                writer.add_scalar("losses/l2", l2, global_step)

                # relabel replay buffer with updated reward networks
                rb.relabel(reward_networks, device, args.batch_size)

        ### ACTION ###
        # choose random action
        if global_step < args.random_exploration_timesteps:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        # policy/actor chooses action
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # execute action
        next_obs, env_rewards, terminations, truncations, infos = envs.step(actions)
        # Get the full state of the Mujoco Model
        full_state = (envs.envs[0].unwrapped.data.qpos.copy(), envs.envs[0].unwrapped.data.qvel.copy())

        # in remaining exploration phase, calculate intrinsic reward
        if args.random_exploration_timesteps <= global_step < args.reward_learning_starts:
            int_rew_calc.add_state(obs)
            model_rewards = int_rew_calc.compute_intrinsic_reward(obs)
        # in reward learning phase, calculate reward based on reward model
        else:
            action = torch.tensor(actions, device=device, dtype=torch.float32)
            state = torch.tensor(obs, device=device, dtype=torch.float32)
            model_rewards = []
            with torch.no_grad():
                for reward_model in reward_networks:
                    model_reward = reward_model.forward(action=action, observation=state)
                    model_rewards.append(model_reward.cpu().numpy())
            model_rewards = np.mean(model_rewards, axis=0)
        episodic_model_rewards += model_rewards

        # Logging and processing of terminations
        if "episode" in infos:
            episode_info = infos["episode"]
            print(f"global_step={global_step}, episodic_return={episode_info['r'][0]}")
            writer.add_scalar("charts/episodic_env_return", episode_info['r'][0], global_step)
            writer.add_scalar("charts/episodic_length", episode_info['l'][0], global_step)
            writer.add_scalar("charts/episodic_model_return", episodic_model_rewards, global_step)
            episodic_model_rewards = 0

        real_next_obs = next_obs.copy()

        # Processing truncations
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = next_obs[idx]

        # Adding step results to Replay Buffer
        rb.add(obs, real_next_obs, actions, env_rewards, model_rewards, terminations, infos, full_state)
        obs = next_obs


        ### ACTOR-CRITIC TRAINING ###
        if global_step >= args.random_exploration_timesteps:
            # sample random minibatch
            data = rb.sample(args.replay_batch_size)

            ### CRITIC TRAINING ###
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.model_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # Update critic / q-network
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            ### ACTOR TRAINING ###
            if global_step % args.policy_update_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_update_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    # Update actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Update entropy coefficient
                    if args.automatic_entropy_coefficient_tuning:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # Update target network
            if global_step % args.target_network_update_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.target_smoothing_coefficient * param.data + (1 - args.target_smoothing_coefficient) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.target_smoothing_coefficient * param.data + (1 - args.target_smoothing_coefficient) * target_param.data)


            # Logging with Tensorflow
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.automatic_entropy_coefficient_tuning:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()