import gymnasium as gym


def make_env(env_id, seed, idx, capture_video, run_name):
    # optional: video capture
    def thunk():
        # if capture_video=True + env_id = 0 (?)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.experimental.wrappers.RecordVideoV0(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def initialize_env(env_id, seed, capture_video, run_name):
    # Create environment
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, capture_video, run_name)])
    # Check if env is continuous
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs