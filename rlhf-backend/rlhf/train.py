import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from render import render_trajectory_gym

def train(envs, rb, actor, reward_network, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, preference_optimizer, args, writer, device, sampler, preference_buffer):

    # [Optional] automatic adjustment of the entropy coefficient
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    ### POLICY LEARNING ### (6)
    for global_step in range(args.total_timesteps):
        ### EXPLORATION PHASE ### (4)
        # TODO Replace Exploration Phase
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        ### REWARD LEARNING ### (7)
        if global_step > args.learning_starts:
            # (8)
            if global_step % args.feedback_frequency == 0: #Is it a good idea to do sampling only with feedback query?
                # (9)
                for _ in range(args.query_size):
                    # (10)
                    (trajectory1, trajectory2) = sampler.uniform_trajectory_pair(args.query_length, args.feedback_frequency)
                    # (11)
                    if args.synthetic_feedback:
                        if sampler.sum_rewards(trajectory1) > sampler.sum_rewards(trajectory2):
                            preference = [1, 0]
                        elif sampler.sum_rewards(trajectory1) < sampler.sum_rewards(trajectory2):
                            preference = [0, 1]
                        else:
                            preference = [0.5, 0.5]
                    else:
                        try:

                            render_trajectory_gym(args.env_id, trajectory1, global_step,"trajectory1")
                            render_trajectory_gym(args.env_id, trajectory2, global_step, "trajectory2")
                            user_input = int(input("Prefer 0, 1, or 2 (0.5/0.5 split): "))
                            if user_input == 0:
                                preference = [1, 0]
                                break
                            elif user_input == 1:
                                preference = [0, 1]
                                break
                            elif user_input == 2:
                                preference = [0.5, 0.5]
                                break
                            else:
                                print("Invalid input. Please enter 0, 1, or 2.")
                        except ValueError:
                            print("Invalid input. Please enter a valid integer (0, 1, or 2).")
                    # (12)
                    preference_buffer.add((trajectory1, trajectory2), preference)

            # (14)
            if global_step % args.feedback_frequency == 0:
                # (15)
                data = preference_buffer.sample(args.pref_batch_size)
                # (16)
                preference_optimizer.train_reward_model(data)
                # (18)
                # TODO Relabel entire Replay Buffer using the updated reward model

        #actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        #actions = actions.detach().cpu().numpy()

        # (21)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions) ##need to change this reward

        # Logging and processing of terminations
        if "episode" in infos:
            episode_info = infos["episode"]
            print(f"global_step={global_step}, episodic_return={episode_info['r'][0]}")
            writer.add_scalar("charts/episodic_return", episode_info['r'][0], global_step)
            writer.add_scalar("charts/episodic_length", episode_info['l'][0], global_step)

        # (21)
        real_next_obs = next_obs.copy()

        # Processing truncations
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = next_obs[idx]

        # Adding to Replay Buffer (22)
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Sample random minibatch (25)
        # TODO: if-statement might be redundant after exploration phase rework
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model (26)
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Policy-Update
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # Target network update
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Logging Tensorflow
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
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        envs.close()
        writer.close()