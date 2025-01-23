from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from .routes import init_routes  # Import routes

def create_app(run_name, preference_buffer, video_queue, stored_pairs, feedback_event, preference_mutex):
    app = Flask(__name__) # Create the Flask app
    CORS(app) # Enable CORS
    socketio = SocketIO(app, cors_allowed_origins="*")

    # Set up the routes and get the notify function
    notify = init_routes(app, socketio, run_name, preference_buffer, video_queue, stored_pairs, feedback_event,
                         preference_mutex)

    return app, socketio, notify  # Return the Flask app, SocketIO, and the notify function