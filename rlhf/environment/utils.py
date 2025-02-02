import gymnasium as gym

def make_env(env_id, seed, idx, run_name,):

    # optional: video capture
    def thunk():
        if "pendulum" in env_id.lower():
            # TODO remove this and implement the trajectory.infos logic in gym_rendering
            env = gym.make(env_id,render_mode="rgb_array")
        else:
            env = gym.make(env_id, render_mode="rgb_array",exclude_current_positions_from_observation=False)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def initialize_env(env_id, seed, run_name):
    # Create environment
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, run_name)])
    # Check if env is continuous
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs