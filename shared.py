# shared.py
import threading
import queue

video_queue = queue.Queue()
preference_mutex = threading.Lock()
stored_pairs = []
feedback_event = threading.Event()