from dataclasses import dataclass

# Passed arguments
@dataclass
class Args:
    exp_name: str = "rlhf"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    num_models: int = 3 #
    """how many reward-models to use (has to be minimum 1)"""
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
    env_id: str = "Hopper-v5"
    """the environment id of the task"""
    total_timesteps: int = int(1e6)
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
    reward_learning_starts: int = 1000
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
    feedback_frequency: int = 1000
    """how often we ask for feedback / update the model (needs to be less or equal to reward_learning_starts)""" # TODO fix this
    uniform_query_size: int = 60
    """how much uniform feedback each iteration"""
    ensemble_query_size: int = 40
    """how much ensemble-based sampling each iteration (needs to be less or equal to uniform [equal = inefficient uniform sampling])"""
    traj_length: int = 90
    """length of trajectories"""
    pref_batch_size: int = 30
    """the batch size of sample from the preference memory"""
    synthetic_feedback: bool = True
    """Toggle synthetic/human feedback"""
    ensemble_sampling: bool = True
    """Toggle ensemble/uniform-based sampling"""
    pretrain_timesteps: int = 500 #TODO: 0 throws error
    """how many steps for random exploration"""
    batch_processing: bool = True # TODO: remove later, not needed


    # Eval Args
    eval_env_id: str = env_id
    eval_max_steps: int = 10000
    n_eval_episodes: int = 1000
    #seed: int = 3 TODO: reasonable seeding

# TODO: We need to add a function to ensure that all args are compatible
# TODO: Needs better structure and documentation, ambigious as is