from dataclasses import dataclass

# Passed arguments
@dataclass
class Args:
    exp_name: str = "rlhf"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    record_every_th_episode: int = 20
    """will record videos every `record_every_th_episode` episodes"""

    # Algorithm specific arguments
    env_id: str = "Walker2d-v5"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 100
    """the batch size of sample from the replay memory"""
    reward_learning_starts: int = 500 #5e3
    """timestep to start learning"""
    reward_model_lr: float = 1e-3
    """the learning rate of the reward model optimizer"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    feedback_frequency: int = 100
    """how often we ask for feedback"""
    query_size: int = 10
    """how much feedback each iteration"""
    query_length: int = 15 #try with 300 later  ## Torch Nan error when over 500 when Hopper, when walker 2d same error by 100 (deterministic, happens every time)
    """length of trajectories"""
    pref_batch_size: int = 50  # Unused arg
    """the batch size of sample from the preference memory"""
    synthetic_feedback: bool = True
    pretrain_timesteps: int = 1000
    """how many steps for random exploration"""


    # Eval Args
    eval_env_id: str = env_id
    eval_max_steps: int = 10000
    n_eval_episodes: int = 1000
    #seed: int = 3 TODO: reasonable seeding

# TODO: We need to add a function to ensure that all args are compatible
# TODO: Needs better documentation, ambigious as is