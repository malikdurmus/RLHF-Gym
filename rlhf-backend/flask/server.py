import os
from urllib import request
from flask import Flask, send_from_directory, jsonify, render_template
from flask_cors import CORS

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

    return send_from_directory(os.path.dirname(video_path), os.path.basename(video_path))

if __name__ == "__main__":
    app.run(debug=True)
