## Initializing networks & buffers
For our algorithm we need different kinds of networks. Firstly, the ones specified in SAC (CleanRL),
And additional ones that make up PEBBLE.

+ SAC: actor network, q1 and q2 (two critic networks)
+ PEBBLE: reward networks(multiple or one, depending on if we do uniform or ensemble based sampling)

A variety of buffers to handle trajectories (state/action)
+ ReplayBuffer (stores the normal state/action next state... given by the environment after taking a step)
Form: (obs, real_next_obs, actions, rewards, true_rewards, terminations, infos)
+ PreferenceBuffer (stores a trajectory pair and a preference)
  Form: [(traj1,traj2), preference]


## Algorithm logic
The PEBBLE algorithm consists of two main parts that make up the training: 
1. Unsupervised Exploration (Random Exploration)
2. Reward model training with Relabeling

In our framework, before we jump into the Unsupervised Exploration, we first fill our ReplayBuffer with random
state/actions. This means, in our very first phase of our algorithm, we have:
+ the argument: pretrain_timesteps, which specifies, how many state/actions we save into our ReplayBuffer

No training is happening yet. After the pretrain_timesteps, we start the second phase, which is the Unsupervised Phase.

## Unsupervised Exploration
In our second phase, we don't use the sample() method from the environment, instead, we let our untrained
actor decide actions. We then take that action and based on the state, we now compute an intrinsic
reward, based on the previous states we visited (during pretraining). We add this reward to our ReplayBuffer
(in our code: model_rewards) and train our actor/critics networks with it. We repeat these steps until we
have an actor that focuses on entropy maximisation. After the global step reached unsupervised_timesteps,
we start with our reward training with Relabeling.

## Reward model training with Relabeling
We start by sampling our trajectory pairs out of our Replaybuffer, either with uniform-(every state/action has the same probability
to get picked) or with ensemble-based-sampling(multiple reward networks choose the state/actions that 
differ the most in terms of computed probability). Assuming we use human feedback, the trajectory pairs
that got sampled and rendered, appear in the frontend where we can give a preference(Form: [preftraj1,preftraj2]).
The trajectory pairs and preference(Form: [(traj1,traj2), preference]) get added to our Preferencebuffer.
We sample out of our Preferencebuffer and train our reward networks with the preferences.
Then, we relabel our entire Replaybuffer according to the newly updated reward_networks.
After this is done, we proceed to:
1. Get an action from our actor network
2. Take that action
3. Calculate a model reward based on that action (if ensemble: avg. of our reward networks)
4. Add it to our Replaybuffer, sample and train the actor/critic networks





