## Initializing Networks & Buffers  

To implement our algorithm, we require multiple neural networks and buffer structures. These components come from **SAC (Soft Actor-Critic)**, **PEBBLE (Preference-Based Learning)**, and **SURF (Semi-Supervised Reward Learning with Data Augmentation)**.  

---

### **Networks**  

#### **SAC (Soft Actor-Critic)**
The SAC framework includes the following networks:  
- **Actor Network:** The policy network that outputs actions.  
- **Q-Networks:** Two critic networks (**q1** and **q2**) used for value estimation.  

#### **PEBBLE / SURF**  
These frameworks introduce additional **reward networks** to model the reward function based on human or synthetic preferences:  
- **Reward Networks:**  
  - Can be either a **single network** or an **ensemble of multiple networks**, depending on whether **uniform sampling** or **ensemble-based sampling** is used.  
  - **PEBBLE** trains the reward networks using **human preference feedback** or **synthetic labels**.  
  - **SURF** enhances PEBBLE with **semi-supervised learning (SSL) and Temporal Data Augmentation (TDA)** to improve feedback efficiency.

---


### **Buffers**  

To store experience and preference data, we use the following buffer structures:  

#### **ReplayBuffer**  
- Stores the standard **state-action transitions** collected from the environment.  
- Used for training the actor-critic networks.  
- **Form:**  
  ```python
  (obs, real_next_obs, actions, rewards, true_rewards, terminations, infos, full_states)
  
#### PreferenceBuffer  

The **PreferenceBuffer** stores **pairs of trajectories** along with **preference labels**, which are used to train the reward model. Preferences can be obtained through:  

- Stores **pairs of trajectories** along with **preference labels**.
- Preference labels are obtained either from **human feedback** or **pseudo-labeling** (in SURF).

- Used to train the **reward model**.
- **Form:**  
  ```python
  [(traj1, traj2), preference]
  
## Algorithm Logic

The **PEBBLE** algorithm consists of two main parts that structure the training process:
1. **Pre-training**
2. **Unsupervised Exploration (Random Exploration)**
3. **Reward Model Training with Relabeling (Preference-based Learning)**

---

## **Phase 1: Pretraining**  

Before training begins, the agent must first **populate the ReplayBuffer** with random state/action pairs. This ensures that the algorithm starts with a diverse set of experiences, providing a foundation for meaningful exploration.

### **Pretraining Process**
- The **ReplayBuffer** is filled with random **state-action transitions** from the environment.
- This phase is controlled by the argument **`pretrain_timesteps`**, which determines how many state-action pairs should be stored before training begins.
- **No network training takes place** during this phase. The primary goal is to collect enough diverse experiences to bootstrap exploration effectively.

Once the **pretrain_timesteps** threshold is reached, the algorithm transitions into **Phase 2: Unsupervised Exploration**, where the actor begins selecting actions based on learned policies rather than random sampling.






## Phase 2: Unsupervised Exploration

After the pretraining phase, the algorithm transitions into the **unsupervised exploration** phase. During this stage, the agent explores the environment without any human feedback, relying on intrinsic rewards to guide its learning.

### **Unsupervised Exploration Process**
1. The **untrained actor** begins choosing actions instead of using random sampling.
2. After executing an action, an **intrinsic reward** is assigned, computed based on the states visited during pretraining.
3. This **intrinsic reward** (stored as `model_rewards` in the code) is added to the **ReplayBuffer**.
4. The **actor-critic networks (SAC components)** are trained using this data, with an emphasis on **entropy maximization**, promoting broader exploration of the environment.
5. When the global step reaches `unsupervised_timesteps`, the algorithm **transitions to reward model training with relabeling**, where feedback-based learning begins.

This phase helps the agent **efficiently explore the environment** (entropy maximization) before incorporating external feedback, ensuring a diverse state-action distribution for later preference-based learning. 🚀  


## Phase 3: Reward Model Training with Relabeling (PEBBLE + SURF)

Once the unsupervised exploration phase is complete, the algorithm transitions to training the reward model. This phase involves preference-based learning, where feedback is used to refine the reward function that guides the policy.

### **Sampling Trajectory Pairs**  
Trajectory pairs are sampled from the ReplayBuffer using one of the following methods:

- **Uniform-based Sampling:** Every state/action pair has an equal chance of being selected.  
- **Ensemble-based Sampling:** Multiple reward networks evaluate trajectories and select those with the highest disagreement in reward predictions. This ensures feedback is provided on the most uncertain or informative samples.  

### **Collecting Feedback (PEBBLE + SURF Enhancements)**  
- For **human-in-the-loop feedback**, the sampled trajectory pairs are rendered in the frontend, where a user selects a preferred trajectory.  
- The labeled trajectory pairs, along with their preferences, are stored in the **PreferenceBuffer**.  

#### **SURF (Semi-Supervised Reward Learning with Data Augmentation) Enhancements:**  
- In addition to human feedback, **SURF** leverages **semi-supervised learning (SSL)** to generate **pseudo-labels** for unlabeled trajectories.  
- **Confidence filtering** ensures only high-confidence pseudo-labels contribute to training.  
- **Temporal Data Augmentation (TDA)** applies random cropping to trajectory segments, improving generalization and feedback efficiency.  

### **Training the Reward Model**  
- The **reward networks** are trained using data from the **PreferenceBuffer**.  
- If using **SURF**, the training loss consists of:  
  - A **supervised loss** from human-labeled preference data.  
  - A **semi-supervised loss** from high-confidence pseudo-labels, weighted by a hyperparameter.  
- After training, the **entire ReplayBuffer is relabeled** using the updated reward networks.  

### **Continuing Actor-Critic Training**  
Once the **ReplayBuffer** is relabeled, the actor network resumes training with the updated reward function:

1. The **actor selects an action**.  
2. The **environment returns the next state**.  
3. If ensemble active: The **reward is computed** as the **average output from the ensemble of reward networks**.  
4. The **new data is added to the ReplayBuffer**.  
5. The **SAC actor and critic networks continue to train** using these updated rewards.  

This phase iteratively refines the reward model, reducing reliance on human feedback while ensuring high-quality reward labels through **semi-supervised learning and data augmentation**. 🚀  


