import random
import numpy as np

class Trajectories:
    def __init__(self, s_dim, a_dim, max_size = 1000):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.max_size = max_size
        self.inputs = []
        self.targets = []

    def add_traj(self, state, action, reward, done):
        sa_array = np.concatenate([state, action], axis=-1)
        r_array = np.array(reward)

        flat_input = sa_array.reshape(1, self.s_dim + self.a_dim)   # Convert to 2D matrix
        flat_target = r_array.reshape(1, 1)                         #


        if len(self.inputs) == 0:  # initital trajectory
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:  # end of trajectory
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs.pop(0)
                self.targets.pop(0)
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0: # new trajectory -> initialize
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else: # append to trajectory
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

    def sample_trajectories(self, target_length=None):
        # filter minimum length
        valid_trajectories = [traj for traj in self.inputs if len(traj) >= target_length]

        # check if at least two valid ones
        if len(valid_trajectories) < 2:
            raise ValueError("Not enough valid trajectories")

        # choose uniform
        traj1, traj2 = random.sample(valid_trajectories, 2)

        # same length
        def adjust_traj(traj, target_length):
            max_start_index = len(traj) - target_length
            start_index = np.random.randint(0, max_start_index + 1)
            return traj[start_index:start_index + target_length]

        adjusted_traj1 = adjust_traj(traj1, target_length)
        adjusted_traj2 = adjust_traj(traj2, target_length)

        return adjusted_traj1, adjusted_traj2



# TEST FUNCTION, REMOVE EVENTUALLY
    def test_traj(self):
        print(f"Number of trajectories: {len(self.inputs)}")
        for i, (inputs, targets) in enumerate(zip(self.inputs, self.targets)):
            print(f"Trajectory {i + 1}:")
            print(f"I: {inputs.shape}, T: {targets.shape}")
            print(f"Length: {len(inputs)}")
            print(f"I: {inputs}, T: {targets}")
