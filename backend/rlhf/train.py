import uuid
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from backend.rlhf.render import render_trajectory_gym


## Helper Functions

def handle_feedback(args, global_step, video_queue, stored_pairs, preference_buffer,
                    feedback_event, notify, trajectory_pairs):

    if not args.synthetic_feedback: # human feedback
        for query in range(len(trajectory_pairs)):
            trajectory_pair = trajectory_pairs[query]
            render_and_queue_trajectories(args,query, global_step, video_queue, stored_pairs,trajectory_pair)

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
            traj_pair = trajectory_pairs[query]
            handle_synthetic_feedback(preference_buffer,traj_pair)


def render_and_queue_trajectories(args,query, global_step, video_queue, stored_pairs,trajectory_pair):

    trajectory1, trajectory2 = trajectory_pair

    # Notify that rendering has started
    print(f"Requested rendering for query {query} at step {global_step}")
    video1_path = render_trajectory_gym(
        args.env_id, trajectory1, global_step, "trajectory1", query)
    video2_path = render_trajectory_gym(
        args.env_id, trajectory2, global_step, "trajectory2", query)

    pair_id = str(uuid.uuid4())  # UUID generation
    video_queue.put((pair_id, trajectory1, trajectory2, video1_path, video2_path))
    stored_pairs.append({
        'id': pair_id,
        'trajectory1': trajectory1,
        'trajectory2': trajectory2
    })

def handle_synthetic_feedback(preference_buffer,trajectory_pair):

    trajectory1, trajectory2 = trajectory_pair

    # print(f"Requested rendering for query {query} at step {global_step} --synthetic")
    # render_trajectory_gym(args.env_id, trajectory1, global_step, "trajectory1", query) # not needed for synthetic feedback
    # render_trajectory_gym(args.env_id, trajectory2, global_step, "trajectory2", query)

    if sum_rewards(trajectory1) > sum_rewards(trajectory2):
        preference = 1
    elif sum_rewards(trajectory1) < sum_rewards(trajectory2):
        preference = 0
    else:
        preference = 0.5
    preference_buffer.add((trajectory1, trajectory2), preference)

def sum_rewards(traj):
    return traj.env_rewards.sum().item()


def train(envs, rb, actor, reward_networks, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer,
          preference_optimizer, args, writer, device, sampler,
          preference_buffer,video_queue,stored_pairs,feedback_event,int_rew_calc, notify, preference_mutex ):

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
    episodic_model_rewards = 0


    ### PEBBLE ALGO ###
    for global_step in range(args.total_timesteps):
        ### REWARD LEARNING ###
        if global_step >= args.unsupervised_timesteps:
            ### FEEDBACK QUERY ###
            if global_step % args.feedback_frequency == 0 or global_step == args.unsupervised_timesteps:
                # ensemble-sampling
                if args.ensemble_sampling:
                    trajectory_pairs = sampler.ensemble_sampling(args.ensemble_query_size, args.uniform_query_size,
                                                                     args.traj_length, args.feedback_frequency,
                                                                     args.synthetic_feedback, preference_optimizer)
                # uniform-sampling
                else:
                    trajectory_pairs = sampler.uniform_trajectory_pair(args.traj_length, args.feedback_frequency)

                # handle feedback
                handle_feedback(args, global_step, video_queue, stored_pairs, preference_buffer,
                    feedback_event, notify, trajectory_pairs)

                ### REWARD MODEL TRAINING ###
                with preference_mutex:
                    entropy_loss, ratio = preference_optimizer.train_reward_models(preference_buffer, args.pref_batch_size)

                writer.add_scalar("losses/entropy_loss", entropy_loss, global_step)
                writer.add_scalar("losses/ratio", ratio, global_step)

                # relabel replay buffer with updated reward networks
                rb.relabel(reward_networks, device)


        ### ACTION ###
        # choose random action
        if global_step < args.pretrain_timesteps:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            int_rew_calc.add_state(obs)
        # policy/actor chooses action
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # execute action
        next_obs, env_rewards, terminations, truncations, infos = envs.step(actions)

        # in exploration phase, calculate intrinsic reward
        if args.pretrain_timesteps <= global_step < args.unsupervised_timesteps:
            model_rewards = int_rew_calc.compute_intrinsic_reward(obs)
        # in reward learning phase, calculate reward based on reward model
        else:
            action = torch.tensor(actions, device=device, dtype=torch.float32)
            state = torch.tensor(obs, device=device, dtype=torch.float32)
            model_rewards = []
            with torch.no_grad():
                for reward_model in reward_networks:
                    model_reward = reward_model.forward(action=action, observation=state) #TODO: ???
                    model_rewards.append(model_reward.cpu().numpy())
            model_rewards = np.mean(model_rewards, axis=0)
        episodic_model_rewards += model_rewards

        # Logging and processing of terminations
        if "episode" in infos:
            episode_info = infos["episode"]
            print(f"global_step={global_step}, episodic_return={episode_info['r'][0]}")
            writer.add_scalar("charts/episodic_return", episode_info['r'][0], global_step)
            writer.add_scalar("charts/episodic_length", episode_info['l'][0], global_step)
            writer.add_scalar("charts/episodic_true_return", episodic_model_rewards, global_step)
            episodic_model_rewards = 0

        real_next_obs = next_obs.copy()

        # Processing truncations
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = next_obs[idx]

        # Adding step results to Replay Buffer
        rb.add(obs, real_next_obs, actions, env_rewards, model_rewards, terminations, infos)
        obs = next_obs


        ### ACTOR-CRITIC TRAINING ###
        if global_step >= args.pretrain_timesteps:
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
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_frequency
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
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # Update target network
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


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
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()