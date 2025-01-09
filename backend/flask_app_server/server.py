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

VIDEO_DIRECTORY = "videos"
video_queue = queue.Queue()
feedback_event = Event()
received_feedback = []
stored_pairs = []

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

    @app.route('/submit_preferences', methods=['POST'])
    def submit_preferences():
        print("Reached here")  # Add this for visibility in console
        global received_feedback
        feedback = request.json.get('preferences', [])
        for feedback_item in feedback:
            pair_id = feedback_item['id']
            preference = feedback_item['preference']
            trajectory_data = next((item for item in stored_pairs if item['id'] == pair_id), None)
            if trajectory_data:
                received_feedback.append((trajectory_data['trajectory1'], trajectory_data['trajectory2'], preference))
        feedback_event.set()
        print(received_feedback)
        return jsonify({'status': 'success'})

    return app, socketio

if __name__ == "__main__":
    app, socketio = create_app()
    logging.basicConfig(level=logging.DEBUG)
    os.makedirs(VIDEO_DIRECTORY, exist_ok=True)

    # RLHF setup
    args = tyro.cli(Args)

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

    preference_buffer = PreferenceBuffer(args.buffer_size, device)

    sampler = TrajectorySampler(rb)

    int_rew_calc = IntrinsicRewardCalculator(k=5)

    # Start training in a separate thread
    training_thread = threading.Thread(
        target=train,
        args=(
            envs, rb, actor, reward_network, qf1, qf2, qf1_target, qf2_target, q_optimizer,
            actor_optimizer, preference_optimizer, args, writer, device, sampler,
            preference_buffer, video_queue, stored_pairs, received_feedback, feedback_event,
            int_rew_calc
        )
    )
    training_thread.daemon = True
    training_thread.start()

    # Run the Flask server
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
