import numpy as np

class PreferenceBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, trajectories, preference):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append([trajectories, preference])

    def sample(self, batch_size, replace=False):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=replace)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)
