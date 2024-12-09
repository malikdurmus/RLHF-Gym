import gymnasium as gym

# Erstellung der Umgebung
def make_env(env_id, seed, idx, capture_video, run_name):
    # Optionale Videoaufzeichnung
    def thunk():
        # Falls arg=True und nur eine Umgebung (?)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk