from dataclasses import dataclass

# Passed arguments
@dataclass
class Args:
    exp_name: str = "rlhf"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    num_models: int = 5
    """how many reward-models to use"""
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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    record_every_th_episode: int = 20
    """will record videos every `record_every_th_episode` episodes"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v5"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    l2: float = 0.01
    """regularization coefficient"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    reward_learning_starts: int = 5e3
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
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    reward_frequency: int = 5000
    """how often we ask for feedback / update the model"""
    uniform_query_size: int = 100
    """how much uniform feedback each iteration"""
    ensemble_query_size: int = 60
    """how much ensemble-based sampling each iteration (needs to be less than uniform)"""
    query_length: int = 90
    """length of trajectories"""
    pretrain_timesteps: int = 1000
    """how many steps for random exploration"""
    feedback_mode: str = "synthetic"
    """the feedback mode, either 'synthetic' for synthetic feedback or 'human' for human feedback"""