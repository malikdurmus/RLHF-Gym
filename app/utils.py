import threading
import os
import tyro
import queue
from rlhf.args import Args

# Parse command-line arguments and store them using tyro
args = tyro.cli(Args)

# Set up directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the absolute path of the directory
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates") # Directory for the HTML template
STATIC_DIR = os.path.join(BASE_DIR, "static") # Directory for the CSS and JS files

video_queue = queue.Queue() # Thread-safe queue for managing video pairs
preference_mutex = threading.Lock() # Mutex to ensure thread-safe access to the shared resources
stored_pairs = [] # Store the trajectory pairs
feedback_event = threading.Event() # Signal when user feedback is available for processing