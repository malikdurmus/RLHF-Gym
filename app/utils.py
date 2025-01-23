import threading
import os
import tyro
import queue
from rlhf.args import Args

args = tyro.cli(Args)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
video_queue = queue.Queue()
preference_mutex = threading.Lock()
stored_pairs = []
feedback_event = threading.Event()