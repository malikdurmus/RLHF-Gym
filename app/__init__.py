from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from .routes import init_routes  # Import routes

def create_app(run_name, preference_buffer, video_queue, stored_pairs, feedback_event, preference_mutex):
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")

    # Initialize routes and get the notify function
    notify = init_routes(app, socketio, run_name, preference_buffer, video_queue, stored_pairs, feedback_event,
                         preference_mutex)

    return app, socketio, notify  # Return notify here