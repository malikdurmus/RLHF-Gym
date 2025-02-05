from dataclasses import dataclass

@dataclass
class Args:

    # -------------------------
    # General arguments
    # -------------------------

    """
    exp_name (str): experiment name
    env_id (str): environment ID
    seed (int): experiment seed for reproducibility
    batch_size (int): batch size for batch processing, depends on cpu/gpu
    """

    exp_name: str = "RLHF_Agent"
    env_id: str = "Hopper-v5"
    seed: int = 1
    batch_size: int = 1024

    # -------------------------
    # CUDA and WandB arguments
    # -------------------------

    """
    torch_deterministic (str): whether to ensure deterministic behavior by setting `torch.backends.cudnn.deterministic=False` 
    cuda (bool): whether to enable CUDA support by default
    wandb_track (bool): whether to enable WandB tracking by default
    wandb_project_name (str): WandB project name 
    wandb_entity (str): WandB entity name (None = default user account)   
    """

    torch_deterministic: bool = True
    cuda: bool = True
    wandb_track: bool = False
    wandb_project_name: str = "RLHF Agent Training"
    wandb_entity: str = None

    ###-----------------------------
    ### Algorithm specific arguments
    ###-----------------------------

    # -------------------------
    # Network arguments
    # -------------------------

    """
    gamma (float): discount factor
    target_smoothing_coefficient (float): target smoothing coefficient
    l2_regularization_coefficient (float): L2 regularization coefficient
    reward_model_lr (float): learning rate of the reward model optimizer
    policy_lr (float): learning rate of the policy model optimizer
    q_network_lr (float): learning rate of the Q network optimizer
    reward_models (int): number of reward models used in the experiment
    """

    gamma: float = 0.99
    target_smoothing_coefficient: float = 0.005
    l2_regularization_coefficient: float = 0.01
    reward_model_lr: float = 1e-3
    policy_lr: float = 3e-4
    q_network_lr: float = 1e-3
    reward_models: int = 3

    # -------------------------
    # Network training arguments
    # -------------------------

    """
    policy_update_frequency (int): frequency at which the policy network is updated
    target_network_update_frequency (int): frequency at which the target network is updated
    (Due to implementation of Denis Yarats' delay by two steps)
    automatic_entropy_coefficient_tuning (bool): whether to use automatic entropy coefficient tuning
    entropy_regularization_coefficient (float): entropy regularization coefficient
    """

    policy_update_frequency: int = 2
    target_network_update_frequency: int = 1
    automatic_entropy_coefficient_tuning: bool = True
    entropy_regularization_coefficient: float = 0.2

    # -------------------------
    # Buffer arguments
    # -------------------------

    """
    replay_buffer_size (int): size of the replay buffer
    replay_batch_size (int): batch size for sampling from the replay buffer
    preference_buffer_size (int): size of the preference buffer
    preference_batch_size (int): batch size for sampling from the preference buffer
    """

    replay_buffer_size: int = int(1e6)
    replay_batch_size: int = 256
    preference_buffer_size: int = replay_buffer_size
    preference_batch_size: int = 32

    # -------------------------
    # Timestep arguments
    # -------------------------

    """
    total_timesteps (int): total number of timesteps for the experiment
    random_exploration_timesteps (int): number of timesteps for random exploration (phase 0.1)
    reward_learning_starts (int): timestep at which reward learning starts
    """

    total_timesteps: int = int(1e6)
    random_exploration_timesteps: int = 2000
    reward_learning_starts: int = 10000

    # -------------------------
    # Feedback query arguments
    # -------------------------

    """   
    synthetic_feedback (bool): whether to use synthetic or human feedback
    feedback_frequency: how often feedback is requested 
    trajectory_length (int): trajectory length of the segments to be compared
    total_queries (int): number of total feedback over the whole training
    ensemble_ratio (int): percentage (0-100) of feedback queries that should use ensemble-based sampling;
                            the remaining queries will be uniformly sampled
    """

    synthetic_feedback: bool = True
    feedback_frequency: int = 10000
    trajectory_length: int = 90
    total_queries: int = 5000
    ensemble_ratio: int = 75

    """
    k (int): number of nearest neighbors used to compute the intrinsic reward.
    """
    k: int = 5

    """
    surf (bool): Toggle SURF (Semi-Supervised Reward Learning with Data Augmentation)
    tda_active (bool): Toggle Temporal Data Augmentation (TDA), which crops trajectory segments
    ssl (bool): Enable semi-supervised learning (SSL) to use pseudo-labeling for reward learning
    crop (int): Defines the intensity (crop size) for trajectory cropping in TDA. 
                Determines min and max crop lengths dynamically based on trajectory length.
    confidence_threshold (float): Minimum confidence required to accept pseudo-labels in SSL
    loss_weight_ssl (float): Weighting factor for the SSL loss term in the reward learning objective
    """

    surf: bool = True
    tda_active: bool = True
    ssl: bool = True
    crop: int = 5
    confidence_threshold: float = 0.99
    loss_weight_ssl: float = 1.0