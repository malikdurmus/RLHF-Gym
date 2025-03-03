import threading
import queue

video_queue = queue.Queue() # Thread-safe queue for managing video pairs
preference_mutex = threading.Lock() # Mutex to ensure thread-safe access to the shared resources
stored_pairs = [] # Store the trajectory pairs
feedback_event = threading.Event() # Signal when user feedback is available for processing