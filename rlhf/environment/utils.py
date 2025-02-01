import gymnasium as gym

def make_env(env_id, seed, idx, capture_video, run_name, record_every_th_episode):

    # optional: video capture
    def thunk():
        if capture_video and idx == 0:
            # TODO can we always use this, if this is needed for rendering?
            env = gym.make(env_id, render_mode="rgb_array",exclude_current_positions_from_observation=False)

        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def initialize_env(env_id, seed, capture_video, run_name,record_every_th_episode):
    # Create environment
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, capture_video, run_name,record_every_th_episode)])
    # Check if env is continuous
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs