import os
from urllib import request
from flask import Flask, send_from_directory, jsonify, render_template
from flask_cors import CORS
from rlhf.buffer import PreferenceBuffer

VIDEO_DIRECTORY = "videos"
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video', methods=['GET'])
def get_video():
    run_name = request.args.get('run_name')
    episode_num = request.args.get('episode', default=1, type=int)

    if not run_name:
        return jsonify({"error": "run_name is required"}), 400

    video_filename = f"training_episode_{episode_num}.mp4"
    video_path = os.path.join(VIDEO_DIRECTORY, run_name, video_filename)

    if not os.path.exists(video_path):
        return jsonify({"error": f"Video {video_filename} not found"}), 404

    return send_from_directory(os.path.join(VIDEO_DIRECTORY, run_name), video_filename)

@app.route('/train-agent', methods=['POST'])
def train_agent(agent=None):
        feedback = request.json

        preference = feedback.get("preference")
        neutral_option = feedback.get("neutralOption")

        reward_video1 = 1.0 if preference == "option1" else 0.0
        reward_video2 = 1.0 if preference == "option2" else 0.0
        neutral_option = 0.0 if neutral_option == "neutral_option" else 0.0

        state_video1 = "state_video1"
        state_video2 = "state_video2"

        preference_buffer = PreferenceBuffer(buffer_size=1000, device='cpu')
        preference_buffer.add(
            [(state_video1, reward_video1), (state_video2, reward_video2)],
            preference
        )

        response = {"Status:": "Training completed"}

        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)