# PEBBLE RLHF framework (with human/synthetic feedback)

[PEBBLE](https://arxiv.org/abs/2106.05091) is an off-policy RL algorithm that uses Unsupervised Learning and Relabeling to solve
complex RL tasks. 
Our project uses PEBBLE to solve problems in continuous space, specifically, in the [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) gym
environments. 
For the base of our algorithm we used SAC, an off-policy algorithm
implemented and found in the [CleanRL](https://github.com/vwxyzjn/cleanrl) Deep Reinforcement library.

TODO: (Demo Video)
## Features
+ Solve MuJoCo environmental tasks
+ Different sampling methods: ensemble based or uniform based
+ Switching between feedback modes: human feedback or synthetic feedback
+ Easy-to-use UI for the human feedback mode (choose between two trajectory pairs, set preference)

## Get started
Prerequisites:
* Python >=3.7.1,<3.11
* [Poetry 1.2.1+](https://python-poetry.org)

Using the project locally:
```bash
# Clone the repo
git clone https://gitlab2.cip.ifi.lmu.de/plankm/sep-groupb.git
cd sep-groupb

# Install dependencies with poetry
poetry install
```
Running the program in the terminal:
```bash
# CD into server.py folder and start the program
cd backend/flask_app_server

# Starting with default values
# Add arguments to change values
poetry run server.py --total_timesteps 100000
                     --feedback_mode "synthetic"
                     --sampling_mode "ensemble"

```
## Feedback modes
Running the program in "human feedback" mode sends a trajectory pair to the UI, after
giving feedback, wait until the new videos are rendered.

## Authors
This project was developed by: 
- Aleksandar Mijatovic
- Martin Plank
- Thang Long Nguyen
- Tobias Huber
- Yusuf Malik Durmus





