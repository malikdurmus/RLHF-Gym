import numpy as np

class PreferenceBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, trajectories, preference):
        """Function to add trajectory pair + preference
        :param trajectories: trajectory pair
        :param preference: preference between trajectory 0 and trajectory 1 Form: e.g 0 for traj1
        :return: None
        """
        if len(trajectories) != 2:
            raise Exception("More than 2 trajectories")
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append([trajectories, preference])

    def sample(self, batch_size, replace=False):
        """Sample randomly out of Preference buffer
        :param batch_size: how many pairs + preferences we are sampling
        :param replace: =False means that we don't sample the same items
        :return: a list with trajectory pairs and the corresponding preference
        """
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=replace)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def reset(self):    # not used anymore
        self.buffer.clear()
