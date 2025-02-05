import numpy as np

class PreferenceBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, trajectories, preference):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append([trajectories, preference])

    def sample_with_validation_sample(self, batch_size, replace):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=replace)

        split_idx = int(len(indices) * (1 - 1/np.e))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_samples = [self.buffer[i] for i in train_indices]
        val_samples = [self.buffer[i] for i in val_indices]

        return train_samples, val_samples

    def __len__(self):
        return len(self.buffer)
