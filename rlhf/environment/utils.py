import gymnasium as gym

def make_env(env_id, seed):
    """
    Create and return an environment instance based on the provided `env_id` and `seed`.
    :param env_id: The ID of the environment to be created.
    :param seed: The seed value for environment reproducibility.

    :return: gym.Env: The created and initialized environment.
    """
    if "pendulum" in env_id.lower() or "pusher" in env_id.lower():
        env = gym.make(env_id, render_mode="rgb_array",
                       max_episode_steps=1000)
    else:
        env = gym.make(env_id, render_mode="rgb_array", terminate_when_unhealthy=True,
                       max_episode_steps=1000)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)  # Seed the action space for reproducibility
    return env


def initialize_env(env_id, seed):
    """
    Initialize a vectorized environment with a single environment instance.
    :param env_id: The ID of the environment to be created.
    :param seed: The seed value for environment reproducibility.

    :return: gym.vector.SyncVectorEnv: The vectorized environment (with one environment).
    """
    # Create and initialize a vectorized environment (with a single environment)
    envs = gym.vector.SyncVectorEnv([lambda: make_env(env_id, seed)])

    # Check if the environment has a continuous action space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action space is supported."

    return envs


def initialize_eval_env(env_id):
    """
    Initialize the evaluation environment with a fixed seed of 0.
    :param env_id: The ID of the environment to be created.

    :return: gym.Env: The created environment for evaluation.

    """
    # Initialize the environment with a seed of 0 for evaluation purposes
    env = make_env(env_id, seed=0)
    return env
