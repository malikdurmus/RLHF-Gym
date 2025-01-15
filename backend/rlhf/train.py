import uuid
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from backend.rlhf.buffer import relabel_replay_buffer
from backend.rlhf.render import render_trajectory_gym


def train(envs, rb, actor, reward_network, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer,
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
    episodic_true_rewards = 0

    ### PEBBLE ALGO ###
    ### PEBBLE: (6)
    ### Unsupervised (2)
    for global_step in range(args.total_timesteps):
        ### REWARD LEARNING ###
        # PEBBLE: (7)
        if global_step > args.reward_learning_starts:
            # (8)
            if global_step % args.feedback_frequency == 0: #Is it a good idea to do sampling only with feedback query?
                # (9)
                if not args.synthetic_feedback:
                    for query in range(args.query_size):
                        trajectory1, trajectory2 = sampler.uniform_trajectory_pair(args.query_length,
                                                                                   args.feedback_frequency,args.synthetic_feedback)
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

                    if not video_queue.qsize() == args.query_size:
                        raise Exception("queue has more/less entries")
                    elif video_queue.qsize() == args.query_size:
                        notify()

                    # now the video queue is full of video paths and pair ids
                    print("Waiting for user feedback...")
                    feedback_event.wait()  # Wait until feedback , received feedback is populated (contains data) after this point
                    feedback_event.clear()  # Reset the event

                # (14)
                # (15)
                with preference_mutex:
                    data = preference_buffer.sample(args.query_size)
                    entropy_loss = preference_optimizer.train_reward_model(
                        data)
                    preference_buffer.reset()
                    stored_pairs.clear()
                # (16)

                writer.add_scalar("losses/entropy_loss", entropy_loss.mean().item(), global_step)
                # (18)
                relabel_replay_buffer(rb, reward_network, device)
                # Clear Preference Buffer



        ### ACTION ###
        # PEBBLE: (20)
        # Unsupervised: (3)
        # choose random action
        if global_step < args.pretrain_timesteps:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            int_rew_calc.add_state(obs)
        # actor chooses action
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # PEBBLE: (21)
        # Unsupervised: (4)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        ### EXPLORATION PHASE ###
        # Unsupervised: (5)
        if args.pretrain_timesteps <= global_step < args.reward_learning_starts:
            true_rewards = int_rew_calc.compute_intrinsic_reward(obs)
        # PEBBLE: (22)
        else:
            action = torch.tensor(actions, device=device, dtype=torch.float32)
            state = torch.tensor(obs, device=device, dtype=torch.float32)
            with torch.no_grad():
                true_rewards = reward_network.forward(action=action, observation=state).cpu().numpy()
        episodic_true_rewards += true_rewards


        # Logging and processing of terminations
        if "episode" in infos:
            episode_info = infos["episode"]
            print(f"global_step={global_step}, episodic_return={episode_info['r'][0]}")
            writer.add_scalar("charts/episodic_return", episode_info['r'][0], global_step)
            writer.add_scalar("charts/episodic_length", episode_info['l'][0], global_step)
            writer.add_scalar("charts/episodic_true_return", episodic_true_rewards, global_step)
            episodic_true_rewards = 0

        real_next_obs = next_obs.copy()

        # Processing truncations
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = next_obs[idx]

        # Adding to Replay Buffer
        # PEBBLE: (22)
        # Unsupervised: (6)
        rb.add(obs, real_next_obs, actions, rewards, true_rewards, terminations, infos)
        obs = next_obs

        # Sample random minibatch
        # PEBBLE: (25)
        # Unsupervised: (9)
        if global_step >= args.pretrain_timesteps:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.true_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # Optimize Critic
            # PEBBLE: (26)
            # Unsupervised: (10)
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

                    # Optimize Actor:
                    # PEBBLE: (26)
                    # Unsupervised: (10)
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

        # Once unsupervised exploration is finished, we overwrite the intrinsic reward with our model
        if global_step == args.reward_learning_starts:
            relabel_replay_buffer(rb, reward_network, device)

    envs.close()
    writer.close()


def process_video_requests(video_request_queue,video_response_queue):
    """
    Continuously processes video rendering requests.
    looks at the video request queue, renders the video and puts it into response_queue
    """
    while True:
        try:
            if not video_request_queue.empty():
                # Fetch rendering request
                env_name, trajectory, global_step, trajectory_id, query = video_request_queue.get()
                print(f"Processing video: {trajectory_id} query {query} at step {global_step}")
                # Render the video
                filename = render_trajectory_gym(env_name, trajectory, global_step, trajectory_id, query)

                print(f"Video rendered and added to response queue: {filename}")
                # Add the completed video to the response queue
                video_response_queue.put(filename)
        except Exception as e:
            print(f"Error processing video request: {e}")