import queue
import os
import logging
import threading
import time
import random

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from multiprocessing import Event
from backend.rlhf.args import Args
from backend.rlhf.environment import initialize_env
from backend.rlhf.intrinsic_reward import IntrinsicRewardCalculator
from backend.rlhf.networks import initialize_networks
from backend.rlhf.buffer import TrajectorySampler, PreferenceBuffer, CustomReplayBuffer
from backend.rlhf.preference_predictor import PreferencePredictor
from backend.rlhf.train import train
from backend.rlhf.utils.evaluate import evaluate_agent

script_dir = os.path.dirname(__file__)
parent_directory = os.path.abspath(os.path.join(script_dir, '..','..'))
VIDEO_DIRECTORY = os.path.join(parent_directory, 'videos')

args = tyro.cli(Args)

video_queue = queue.Queue()
feedback_event = Event()
preference_buffer = PreferenceBuffer(args.buffer_size)
stored_pairs = []
preference_mutex = threading.Lock()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key_here'
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")

    @app.route("/", methods=["GET"])
    def serve_landing_page():
        return send_from_directory("templates", "index2.html")

    @app.route('/get_video_pairs', methods=["GET"])
    def get_video_pairs():
        video_pairs = []
        while not video_queue.empty():
            pair_id, _, _, video1_path, video2_path = video_queue.get()
            video_pairs.append({'id': pair_id, 'video1': video1_path, 'video2': video2_path})
        return jsonify({'video_pairs': video_pairs})

    @app.route('/videos/<filename>')
    def serve_video(filename):
        return send_from_directory(VIDEO_DIRECTORY, filename)

    @app.route('/submit_preferences', methods=['POST'])
    def submit_preferences():
        global preference_buffer
        feedback = request.json.get('preferences', [])
        print("feedback is er",feedback)
        for feedback_item in feedback:
            pair_id = feedback_item['id']
            preference = feedback_item['preference']
            with preference_mutex:
                trajectory_data = next((item for item in stored_pairs if item['id'] == pair_id), None)
                if trajectory_data:
                    preference_buffer.add( (trajectory_data['trajectory1'], trajectory_data['trajectory2']), preference)

        feedback_event.set() # Signal training thread
        return jsonify({'status': 'success'})

    def notify_new_video_pairs():
        # this notifies the frontend that the videos are ready, (no need for inefficient polling)
        socketio.emit('new_video_pairs', {'status': 'ready'})
    return app, socketio, notify_new_video_pairs

if __name__ == "__main__":
    app, socketio, notify = create_app()
    logging.basicConfig(level=logging.DEBUG)
    os.makedirs(VIDEO_DIRECTORY, exist_ok=True)

    # RLHF setup
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_path = os.path.join(parent_directory, "runs", run_name)

    writer = SummaryWriter(run_path)

    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    #Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = initialize_env(args.env_id, args.seed, args.capture_video, run_name, args.record_every_th_episode)

    actor, reward_network, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer = (
        initialize_networks(envs, device, args.policy_lr, args.q_lr))

    preference_optimizer = PreferencePredictor(reward_network, reward_model_lr=args.reward_model_lr, device=device)

    rb = CustomReplayBuffer.initialize(envs, args.buffer_size, device)

    sampler = TrajectorySampler(rb)

    int_rew_calc = IntrinsicRewardCalculator(k=5)

    # Start training in a separate thread
    training_thread = threading.Thread(
        target=train,
        args=(
            envs, rb, actor, reward_network, qf1, qf2, qf1_target, qf2_target, q_optimizer,
            actor_optimizer, preference_optimizer, args, writer, device, sampler,
            preference_buffer, video_queue, stored_pairs, feedback_event,
            int_rew_calc, notify, preference_mutex
        )
    )
    training_thread.daemon = True

    # evaluation_thread = threading.Thread(
    #     target = evaluate_agent,
    #     args= (args.eval_env_id, args.eval_max_steps,  args.n_eval_episodes , actor ,
    #     device, 0,
    #     args.capture_video, run_name, args.record_every_th_episode
    #     )
    # )

    # Run the Flask server
    training_thread.start()
    if not args.synthetic_feedback: socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
    # If debug is true, the app will render the first batch twice

    # TODO: Needs documentation
    # TODO: The get_video_pairs endpoint shouldnt be consumed (disappear) before the frontend makes the post request # this can be handled either in frontend or backend
