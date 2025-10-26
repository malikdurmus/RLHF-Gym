from flask import request, jsonify, send_from_directory
import os
from .utils import TEMPLATES_DIR

def init_routes(app, socketio, run_name, preference_buffer, video_queue, stored_pairs, feedback_event, preference_mutex):
    @app.route("/", methods=["GET"])
    # Serve the HTML landing page
    def serve_landing_page():
        return send_from_directory("templates", "index2.html")

    # Fetch available video pairs from the video queue
    @app.route('/get_video_pairs', methods=["GET"])
    def get_video_pairs():
        video_pairs = []
        while not video_queue.empty():
            pair_id, _, _, video1_path, video2_path = video_queue.get() #video_queue.get() should not be executed until the queue is full
            video_pairs.append({'id': pair_id, 'video1': video1_path, 'video2': video2_path})
        return jsonify({'video_pairs': video_pairs})

    # Return the current run name
    @app.route("/get_run_name", methods=["GET"])
    def get_run_name():
        return jsonify({'run_name': run_name})

    # Serve the videos files from the specified run directory
    @app.route('/videos/<flask_run_name>/<filename>')
    def serve_video(flask_run_name, filename):
        video_dir = os.path.join("../videos", flask_run_name)
        return send_from_directory(video_dir, filename, mimetype='video/mp4')

    # Handle preferences from the user
    @app.route('/submit_preferences', methods=['POST'])
    def submit_preferences():
        feedback = request.json.get('preferences', [])
        print("Received feedback: ", feedback)
        # Iterate over every user feedback
        for feedback_item in feedback:
            pair_id = feedback_item['id']
            preference = feedback_item['preference']
            with preference_mutex:
                # Find the trajectory data that matches the pair ID
                trajectory_data = next((item for item in stored_pairs if item['id'] == pair_id), None)
                # If matching trajectory data was found, add the user preference to the preference buffer
                if trajectory_data:
                    preference_buffer.add((trajectory_data['trajectory1'], trajectory_data['trajectory2']), preference)

        feedback_event.set()  # Notify the training thread
        return jsonify({'status': 'success'})

    # Notify about new video pairs
    def notify_new_video_pairs():
        socketio.emit('new_video_pairs', {'status': 'ready'})

    return notify_new_video_pairs  # Return the notify function