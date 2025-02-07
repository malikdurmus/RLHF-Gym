# Algorithm Overview

## Initializing the Networks & Buffers  

Our algorithm implementation requires multiple neural networks and buffer structures coming from the following 
frameworks:
+ **SAC** (Soft Actor-Critic)
+ **PEBBLE** (Preference-Based Learning)
+ **SURF** (Semi-Supervised Reward Learning with Data Augmentation)

---

### Networks 

+ **SAC** includes the following networks:  
  + **Actor Network:** The policy network to output actions.
  + **Q-Networks:** Two critic networks (q1 and q2) to estimate values.


+ **PEBBLE** and **SURF** introduce additional reward networks (either a single network or an ensemble of multiple networks, 
depending on if uniform-based sampling or ensemble-based sampling is used) to model the reward function:
  + **PEBBLE**: Trains the reward networks by using human feedback or synthetic labels.  
  + **SURF**: Extends PEBBLE with semi-supervised learning (SSL) and Temporal Data Augmentation (TDA) to improve the 
  feedback efficiency.

---

### Buffers

#### Replay Buffer

The Replay Buffer is used to train the actor-critic networks by storing the standard state-action pairs collected 
from the environment.

+ **Structure:**  
  ```python
  (obs, real_next_obs, actions, rewards, true_rewards, terminations, infos, full_states)
  
#### Preference Buffer

The Preference Buffer stores trajectory pairs along with preference labels, which are used to train the reward model.
These Preference labels are obtained from either human feedback or pseudo-labeling (in SURF).

+ **Structure:**  
  ```python
  [(traj1, traj2), preference]
  
## PEBBLE Algorithm Logic

The PEBBLE algorithm is structured into three phases:
1. **Pretraining**
2. **Unsupervised Exploration** (Random Exploration)
3. **Reward Model Training with Relabeling** (Preference-based Learning)

---

## Phase 1: Pretraining 

In the first phase, the agent fills the Replay Buffer with random state-action pairs, ensuring that the algorithm 
starts with a broad variety of experiences for effective exploration.

Here, the Replay Buffer is filled with random state-action pairs from the environment. 

The argument `pretrain_timesteps` determines the stored number of state-action pairs before the training begins. 
Once the `pretrain_timesteps` limit is reached, the second phase begins.

## Phase 2: Unsupervised Exploration

In the second phase, the agent explores the environment using only intrinsic rewards, without any human feedback:

1. The untrained actor starts selecting actions, rather than relying on random sampling.
2. After taking an action, the agent receives an intrinsic reward, which is computed based on the states
encountered during the pretraining.
3. This intrinsic reward (stored as `model_rewards` in the code) is added to the Replay Buffer.
4. The actor-critic networks are then trained using this data, with a focus on entropy maximization for 
broader environment exploration.
5. Once the global step reaches `unsupervised_timesteps`, the third phase begins.

## Phase 3: Reward Model Training with Relabeling (PEBBLE and SURF)

The third phase includes preference-based learning, where feedback is used to improve the policy reward function.

### Sampling Trajectory Pairs
The trajectory pairs are sampled from the Replay Buffer using one of the following methods:

+ **Uniform-based sampling:** Every state-action pair has an equal probability of being selected.  
+ **Ensemble-based sampling:** Multiple reward networks evaluate and select the trajectories with the most varying
reward predictions. This ensures that feedback is provided on the most uncertain or informative samples.  

### Collecting Feedback (PEBBLE and SURF Enhancements)
For human-in-the-loop feedback, the sampled trajectory pairs are shown in the frontend. 
Here, a user selects a preferred trajectory or chooses the neutral option, if the trajectories are too similar. 
The labeled trajectory pairs, along with their preferences, are then stored in the Preference Buffer.  

##### SURF Enhancements:  
In addition to human feedback, SURF uses SSL to generate pseudo-labels for unlabeled trajectories. 

Here, Confidence filtering ensures that only high-confidence pseudo-labels are used in the training. 

TDA randomly crops trajectory segments to improve the generalization and feedback efficiency.  

### Training the Reward Model 
The reward networks are trained using the Preference Buffer data.  

When SURF is used, the training loss consists of:  
  + A **supervised loss** based on human-labeled preferences 
  + A **semi-supervised loss** using high-confidence pseudo-labels, weighted by a hyperparameter.

After the training, the entire Replay Buffer is relabeled using the updated reward networks.  

### Continuing the Actor-Critic Training  
Once the Replay Buffer is relabeled, the actor network training continues with the updated reward function:

1. The actor selects an action.  
2. The environment returns the next state.
3. If the reward network ensemble is active, the reward is computed as the average reward from this ensemble.
4. This updated data is added to the Replay Buffer.  
5. The SAC actor-critic networks continue the training using these updated data.