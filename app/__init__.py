import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from app.utils import BASE_DIR, TEMPLATES_DIR, STATIC_DIR, video_queue, preference_mutex, stored_pairs, feedback_event

def create_app(run_name, preference_buffer):
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")
    os.makedirs(os.path.join(f"videos/{run_name}"), exist_ok=True)

    @app.route("/", methods=["GET"])
    def serve_landing_page():
        return send_from_directory(TEMPLATES_DIR, "index2.html")

    @app.route('/get_video_pairs', methods=["GET"])
    def get_video_pairs():
        video_pairs = []
        while not video_queue.empty():
            pair_id, _, _, video1_path, video2_path = video_queue.get()
            video_pairs.append({'id': pair_id, 'video1': video1_path, 'video2': video2_path})
        return jsonify({'video_pairs': video_pairs})

    @app.route("/get_run_name", methods=["GET"])
    def get_run_name():
        return jsonify({'global_run_name': run_name})

    @app.route('/videos/<flask_run_name>/<filename>')
    def serve_video(flask_run_name, filename):
        video_dir = os.path.join("../videos", flask_run_name)   # TODO make this an absolute path eventually
        return send_from_directory(video_dir, filename)

    @app.route('/submit_preferences', methods=['POST'])
    def submit_preferences():
        feedback = request.json.get('preferences', [])
        print("Received feedback: ",feedback)
        for feedback_item in feedback:
            pair_id = feedback_item['id']
            preference = feedback_item['preference']
            with preference_mutex:
                trajectory_data = next((item for item in stored_pairs if item['id'] == pair_id), None)
                if trajectory_data:
                    preference_buffer.add((trajectory_data['trajectory1'], trajectory_data['trajectory2']), preference)

        feedback_event.set()  # Signal training thread
        return jsonify({'status': 'success'})

    def notify_new_video_pairs():
        socketio.emit('new_video_pairs', {'status': 'ready'})

    return app, socketio, notify_new_video_pairs, video_queue, stored_pairs, feedback_event, preference_mutex