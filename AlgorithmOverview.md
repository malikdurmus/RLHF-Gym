# Algorithm Overview

## Initializing the Networks and Buffers
**The algorithm requires the networks specified in SAC (as implemented in CleanRL), and extends them with 
additional networks for PEBBLE:**

+ SAC: actor network, q1 and q2 (two critic networks)
+ PEBBLE: reward networks (single or multiple networks, depending on whether uniform-based or ensemble-based 
sampling is used)

**Additionally, the algorithm uses different buffers to manage trajectories (state-action):**

+ **Replay Buffer** (storing standard environment experiences after taking a step)
  + Structure: (obs, real_next_obs, actions, rewards, true_rewards, terminations, infos)
+ **Preference Buffer** (storing a trajectory pair and a preference)
  + Structure: [(traj1,traj2), preference]

## Algorithm Logic
**The PEBBLE algorithm consists of two main parts during the training:** 
1. **Unsupervised Exploration** (Random Exploration)
2. **Reward model training with Relabeling**

In our framework, the Replay Buffer is filled with random state-action pairs before the Unsupervised Exploration. 
Hence, in the first phase of the algorithm, the argument pretrain_timesteps is used (specifying, how many state-actions 
pairs are stored in the  Replay Buffer)

At this stage, no training has happened yet. After the pretrain_timesteps, the second phase is started:
the Unsupervised Phase.

## Unsupervised Exploration
In the second phase, instead of using the  sample() method from the environment, our untrained actor decides its actions.
These actions are executed, and an intrinsic reward is computed based on previously visited states during the pretraining.
This reward is stored in the Replay Buffer (referred to as model_rewards in our code), and the actor-critic networks are
trained with it. These steps are repeated until the actor focuses on entropy maximization.

Once the global step reaches unsupervised_timesteps, the next phase is started: reward training with Relabeling.

## Reward Model Training with Relabeling
This phase starts by sampling the trajectory pairs out of the Replay Buffer by using either one of these sampling methods:
+ Uniform-based sampling , where each state-action pair has an equal probability of being selected
+ Ensemble-based sampling, where multiple reward networks choose the state-action pairs that differ the most in computed
probabilities

If human feedback is used, the sampled and rendered trajectory pairs are displayed in the UI, allowing users to provide
preferences in the form of [preftraj1,preftraj2]. These trajectory pairs and preferences are then stored in the 
Preference Buffer in the form of [(traj1,traj2), preference].
Next, the reward networks are trained using the Preference Buffer preferences. Afterwards, relabel the entire Replay 
Buffer based on the newly updated reward_networks.

Then, the algorithm proceeds with these steps:
1. Get an action from the actor network
2. Take the action
3. A model reward is computed based on the action (for ensemble-based: average of the reward networks)
4. Add the model reward to the Replay Buffer, sample it and train the actor-critic networks





