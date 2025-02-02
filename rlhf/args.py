from dataclasses import dataclass

# Passed arguments
@dataclass
class Args:
    # General arguments
    exp_name: str = "rlhf"
    """the name of this experiment"""
    env_id: str = "Hopper-v5"
    """the environment id of the task"""
    seed: int = 1
    """seed of the experiment"""

    # ... arguments
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RLHF"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True  # TODO do we still need this? see environment.py
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    record_every_th_episode: int = 20   # TODO do we still need this?
    """will record videos every `record_every_th_episode` episodes"""


    ### Algorithm specific arguments ###
    # Network arguments
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    l2: float = 0.01
    """regularization coefficient"""
    reward_model_lr: float = 1e-3
    """the learning rate of the reward model optimizer"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    num_models: int = 4
    """amount of reward models"""

    # Network training arguments
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target networks"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""

    # Buffer arguments
    replay_buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    replay_batch_size: int = 256
    """the batch size of sample from the replay memory"""
    pref_buffer_size: int = replay_buffer_size
    """the preference buffer size"""
    pref_batch_size: int = 30
    """the batch size of sample from the preference memory"""

    # Timestep arguments
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    pretrain_timesteps: int = 1000 #TODO: 0 throws error
    """phase 1: how many steps for random exploration"""
    unsupervised_timesteps: int = 10000
    """phase 2: unsupervised exploration"""
    """
    Remaining timesteps:
    phase 3: reward learning
    """

    # Feedback query arguments
    synthetic_feedback: bool = False
    """Toggle synthetic/human feedback"""
    ensemble_sampling: bool = True
    """Toggle ensemble/uniform-based sampling"""
    feedback_frequency: int = 5000
    """how often we ask for feedback / update the model (needs to be less or equal to reward_learning_starts)""" # TODO fix this
    traj_length: int = 60
    """length of trajectories"""
    uniform_query_size: int = 5
    """how much uniform feedback each iteration"""
    ensemble_query_size: int = 2
    """how much ensemble-based sampling each iteration (needs to be less or equal to uniform [equal = inefficient uniform sampling])"""

    # SSL & TDA Arguments
    surf: bool = True
    min_crop_length: int = 45
    max_crop_length: int = 55
    confidence_threshold: float = 0.99

    # Evaluation arguments
    eval_env_id: str = env_id
    eval_max_steps: int = 10000
    n_eval_episodes: int = 1000
    eval_seed : int = 3

    batch_processing: bool = True # TODO: remove later, not needed


# TODO: We need to add a function to ensure that all args are compatible
# TODO: Needs better structure and documentation, ambigious as is

# if surf=True, then both min length, max length must not be None max length must be < traj length