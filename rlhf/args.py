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
    """

    exp_name: str = "RLHF_agent"
    env_id: str = "Hopper-v5"
    seed: int = 1

    # -------------------------
    # CUDA and WandB arguments
    # -------------------------

    """
    is_torch_deterministic (str): whether to ensure deterministic behavior by setting `torch.backends.cudnn.deterministic=False` 
    enable_cuda (bool): whether to enable CUDA support by default
    wandb_track (bool): whether to enable WandB tracking by default
    wandb_project_name (str): WandB project name 
    wandb_entity (str): WandB entity name (None = default user account)   
    """

    torch_deterministic: bool = True #rename?
    cuda: bool = True # rename?
    track: bool = False #rename?
    wandb_project_name: str = "RLHF Agent Training"
    wandb_entity: str = None


    capture_video: bool = True  # TODO do we still need this? see environment.py @malik
    """whether to capture videos of the agent performances (check out `videos` folder)"""

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
    tau: float = 0.005 #rename?
    l2: float = 0.01  # rename?
    num_models: int = 3 #rename?
    reward_model_lr: float = 3e-4
    policy_lr: float = 3e-4
    q_lr: float = 1e-3 #rename?

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

    policy_frequency: int = 2 #rename?
    target_network_frequency: int = 1 #rename?
    autotune: bool = True #rename?
    alpha: float = 0.2 #rename?

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
    pref_buffer_size: int = replay_buffer_size #rename?
    pref_batch_size: int = 64 #rename?

    # -------------------------
    # Timestep arguments
    # -------------------------

    """
    total_timesteps (int): total number of timesteps for the experiment
    pretraining_timesteps (int): number of timesteps for random exploration (phase 0.1)
    unsupervised_timesteps (int): number of timesteps for unsupervised exploration (phase 0.2)
    """

    total_timesteps: int = int(1e6)
    pretrain_timesteps: int = 2000 #rename?
    unsupervised_timesteps: int = 10000

    # -------------------------
    # Feedback query arguments
    # -------------------------

    """   
    synthetic_feedback (bool): whether to use synthetic or human feedback
    ensemble_sampling (bool): whether to use ensemble-based or uniform-based sampling
    feedback_frequency: how often feedback is requested 
    trajectory_length (int): trajectory length during each feedback iteration
    uniform_query_size (int): number of feedback samples requested uniformly during each feedback iteration
    ensemble_query_size (int): number of ensemble-based feedback samples requested during each feedback iteration
    """

    synthetic_feedback: bool = False
    ensemble_sampling: bool = True
    feedback_frequency: int = 2500
    traj_length: int = 90  #rename?
    uniform_query_size: int = 8
    ensemble_query_size: int = 4

    # TODO Remove all of the below? @malik
    batch_processing: bool = True  # TODO: remove later, not needed
    """
    # Evaluation arguments
    eval_env_id: str = env_id
    eval_max_steps: int = 10000
    n_eval_episodes: int = 1000
    eval_seed : int = 3
    ###
    """

    # TODO rename project-wide @Martin