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
+ Easy-to-use UI for the human feedback mode (choose between two trajectory pairs, set a preference)

## Get started
Prerequisites:
* Python >=3.7.1,<3.11
* [Poetry 1.2.1+](https://python-poetry.org)

## Using the project locally:
```bash
# Clone the repo
git clone https://gitlab2.cip.ifi.lmu.de/plankm/sep-groupb.git
cd sep-groupb

# Install dependencies with poetry
poetry install
```
## Running the program in the terminal:
```bash
# CD into server.py folder and start the program
cd backend/flask_app_server

# Starting with default values
# Add arguments to change values
poetry run server.py --total_timesteps 100000
                     --synthetic_feedback True
                     --ensemble_sampling False

```
## Feedback modes
Running the program with synthetic_feedback = False, sends a trajectory pair to the UI. After
giving feedback, wait until the new videos are rendered.

## Authors
This project was developed by: 
- Aleksandar Mijatovic
- Martin Plank
- Thang Long Nguyen
- Tobias Huber
- Yusuf Malik Durmus

## Citation
As mentioned before, we used an implementation of SAC which can be found in the CleanRL Deep Reinforcement library.

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```




