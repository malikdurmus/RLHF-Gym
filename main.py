import os
import time
import random
import numpy as np
import torch
import tyro
import threading
from torch.utils.tensorboard import SummaryWriter
from app import create_app
from rlhf.args import Args
from rlhf.environment.utils import initialize_env
from rlhf.networks.utils import initialize_networks
from rlhf.rendering.gym_renderer import record_video
from rlhf.training.pebble import train
from rlhf.buffers import *
from rlhf.training.reward_learning import PreferencePredictor
from rlhf.training.intrinsic_reward import IntrinsicRewardCalculator
from shared import video_queue, preference_mutex, stored_pairs, feedback_event

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    os.makedirs(os.path.join(f"videos/{run_name}"), exist_ok=True)
    os.makedirs(os.path.join(f"evaluation/{run_name}"), exist_ok=True)


    # Initialize writer
    writer = SummaryWriter(f"runs/{run_name}")

    # Add hyperparameters to writer
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # Seeding (for reproduction)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Choose hardware
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialize environment (environment.py)
    envs = initialize_env(args.env_id, args.seed, run_name)

    # Initialize networks (networks.py)
    actor, reward_networks, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer = (
        initialize_networks(envs, device, args.policy_lr, args.q_lr, args.num_models))

    # Initialize preference predictor (preference_predictor.py)
    preference_optimizer = PreferencePredictor(reward_networks, reward_model_lr=args.reward_model_lr, device=device, l2=args.l2)

    # Initialize preference buffer (buffer.py)
    preference_buffer = PreferenceBuffer(args.pref_buffer_size)

    # Initialize replay buffer (buffer.py)
    rb = CustomReplayBuffer.initialize(envs, args.replay_buffer_size, device)

    # Initialize sampler (buffer.py)
    sampler = TrajectorySampler(rb, device)

    # Initialize intrinsic reward calculator
    int_rew_calc = IntrinsicRewardCalculator(k=5)

    # Create app if human feedback is used
    if not args.synthetic_feedback:
        app, socketio, notify = create_app(run_name, preference_buffer, video_queue, stored_pairs, feedback_event, preference_mutex)
    else:
        app, socketio, notify = None, None, None


    # Create a function to run the Flask app in a separate thread
    def run_flask_app():
        if not args.synthetic_feedback:  # human feedback
            socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)


    # Create an event to signal when training is done
    training_done_event = threading.Event()
    # Start training thread
    def train_thread_func():
        train(
            envs, rb, actor, reward_networks, qf1, qf2, qf1_target, qf2_target, q_optimizer,
            actor_optimizer, preference_optimizer, args, writer, device, sampler,
            preference_buffer, video_queue, stored_pairs, feedback_event,
            int_rew_calc, notify, preference_mutex, run_name
        )
        # Signal that training is done
        training_done_event.set()


    training_thread = threading.Thread(target=train_thread_func)
    training_thread.start()

    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Wait for the training thread to finish before starting the evaluation
    training_thread.join()

    # Once training is done, start the evaluation thread
    evaluation_thread = threading.Thread(
        target=record_video,
        args=(args.env_id, args.seed, f"evaluation/{run_name}", device, actor)
    )
    #evaluation_thread.start()

    # Wait for evaluation thread to finish
    #evaluation_thread.join()
