import time
import random
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from args import Args
from environment import initialize_env
from networks import initialize_networks
from train import train
from buffer import initialize_rb



if __name__ == "__main__":

### SETUP ###

    # Parse arguments (args.py)
    args = tyro.cli(Args)

    # Unique run_name
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Save results
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding (for reproduction)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Choose hardware
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialize environment (environment.py)
    envs = initialize_env(args.env_id, args.seed, args.capture_video, run_name, args.record_every_th_episode)

    # Initialize networks (networks.py)
    actor, reward_network, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, preference_optimizer = initialize_networks(
        envs, device, args.policy_lr, args.q_lr ,args.reward_model_lr
    )

    # Initialize replay buffer (buffer.py)
    rb = initialize_rb(envs, args.buffer_size, device)

    # optional: track weight and biases
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

### TRAINING ### (train.py)

    # start training
    train(
        envs=envs,
        rb=rb,
        actor=actor,
        reward_network=reward_network,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        preference_optimizer=preference_optimizer,
        args=args,
        writer=writer,
        device=device,
    )