# Agent Design for Ms. Pac-Man! 

**1. Possible Actions in Ms. Pac-Man (Gymnasium & ALE)**

When using Gymnasium with the Atari Learning Environment (ALE), the action space can be configured.

*   **`full_action_space=False` (Minimal Action Set - What you are using):**
    *   For Ms. Pac-Man, this typically results in **9 discrete actions**.
    *   These actions usually correspond to:
        1.  `NOOP` (No Operation)
        2.  `UP`
        3.  `DOWN`
        4.  `LEFT`
        5.  `RIGHT`
        6.  `UPRIGHT`
        7.  `UPLEFT`
        8.  `DOWNRIGHT`
        9.  `DOWNLEFT`
    *   Your `config['n_actions'] = 9` and the `AtariEnv` class using `full_action_space=False` align with this. The check in `train_dqn.py` (`if config['n_actions'] != env.action_space.n:`) is good for ensuring consistency.

*   **`full_action_space=True` (Full 18 Actions):**
    *   This would expose all 18 joystick actions (including diagonals and button presses, though the button is irrelevant for Ms. Pac-Man). This generally makes the learning problem harder due to a larger action space and redundant actions. The minimal set is usually preferred for faster learning.

*   **Verification**:
    *   You can always programmatically check the action meanings:
        ```python
        # In your train_dqn.py or a separate script
        # from train_dqn import AtariEnv # Assuming AtariEnv is accessible
        # env = AtariEnv(game_id='ALE/MsPacman-v5')
        # print(f"Action space size: {env.action_space.n}")
        # print(f"Action meanings: {env.get_action_meanings()}")
        ```
    This will give you the definitive list for your specific environment setup.

**2. How to Make Sure We Are Designing the Best Agent for Ms. Pac-Man**

Designing the "best" agent is an iterative process of making informed choices, careful experimentation, and rigorous evaluation. There's no single magic bullet. Here's a deep dive into considerations for Ms. Pac-Man:

**I. Understanding the Game's Core Mechanics & Challenges:**

*   **Objective:** Maximize score by eating pellets, power pellets, fruits, and ghosts (when vulnerable).
*   **Key Entities:** Ms. Pac-Man, 4 ghosts (Blinky, Pinky, Inky, Sue/Clyde - with distinct behaviors/personalities), pellets, power pellets, fruits, tunnels, multiple mazes.
*   **Partial Observability:** While you see the whole screen, predicting exact future ghost movements without internal state can be hard. The ghosts' "modes" (chase, scatter, frightened) are critical. This is where an LSTM *might* help.
*   **Temporal Dependencies:**
    *   Remembering which power pellets have been eaten.
    *   Knowing how long ghosts remain frightened.
    *   Anticipating ghost movements based on their recent paths or modes.
*   **Sparse vs. Dense Rewards:** Ms. Pac-Man has relatively dense rewards (pellets give small, frequent rewards). This makes basic Q-learning feasible.
*   **Exploration:** The agent needs to explore the maze to find pellets and fruits, but also learn to avoid or hunt ghosts strategically.
*   **Generalization:** The agent plays on multiple different maze layouts. It needs to learn general strategies, not just memorize one maze.

**II. Key Design Choices for Your DQN Agent:**

*   **A. State Representation (Input to the Network):**
    *   **Current:** Stack of 8 grayscale frames (84x84). This is a strong standard.
        *   *Stack Depth (8):* Provides motion information and some history. Is 8 optimal? Maybe 4 is enough, or perhaps more could help the LSTM. This is a tunable hyperparameter.
        *   *Grayscale:* Loses color information. For Ms. Pac-Man, color distinguishes ghosts and fruit. It might be beneficial, but grayscale is standard for simplicity and to reduce input dimensionality.
        *   *Resolution (84x84):* Standard from the Nature DQN paper.
    *   **Considerations:** Could more sophisticated features help (e.g., explicit locations of ghosts, Ms. Pac-Man, pellets)? Yes, but this moves away from end-to-end learning from pixels and adds complexity. For now, pixel-based input is the focus.

*   **B. Network Architecture (Your `DQNCNN` or `DQNCNN_LSTM`):**
    *   **Convolutional Base (Shared):**
        *   The current architecture (3 conv layers: 32k8s4, 64k4s2, 64k3s1) is standard (similar to Nature DQN).
        *   *Activation (ReLU in CNN):* Good default.
    *   **Recurrent Layer (LSTM - your current experiment):**
        *   *Purpose:* To capture longer-term temporal dependencies beyond the frame stack.
        *   *Hidden Size (512):* A reasonable starting point. Could be tuned.
        *   *Integration:* Placed after CNN features, before FC layers.
    *   **Fully Connected Layers (Outputting Q-values):**
        *   *Size (512 units):* Common.
        *   *Activation (GELU):* Your current choice. Smoother than ReLU, potentially beneficial.
        *   *Output Layer:* Linear layer with `n_actions` outputs.
    *   **Architectural Variants to Consider (Potentially for future experiments, not now):**
        *   **Dueling DQN:** We discussed this before. It separates the value function (V(s)) and advantage function (A(s,a)). Often helps, especially in games with many actions or where the value of the state is independent of the specific action taken.
        *   **ResNet/Deeper CNNs:** Could potentially extract better features, but at the cost of complexity and training time.

*   **C. Learning Algorithm Components:**
    *   **Double DQN:** You are using this (target network uses policy net's action choice). Helps reduce Q-value overestimation. Keep it.
    *   **Prioritized Experience Replay (PER):** You found it wasn't better for your non-LSTM setup. It's worth re-evaluating with the LSTM, as the "surprising" transitions an LSTM might learn from could be different.
    *   **Loss Function (Huber Loss):** Good choice for robustness.
    *   **Optimizer (Adam):** Good default. Learning rate is critical.
    *   **Discount Factor (Gamma - `0.99`):** Standard, encourages long-term planning.
    *   **Target Network Update Frequency:** A key hyperparameter. Too frequent can lead to instability; too infrequent, and the target is too stale.

*   **D. Exploration Strategy:**
    *   **Epsilon-Greedy:** Your current method.
        *   *Decay Schedule:* Crucial. Decaying too fast means insufficient exploration; too slow means prolonged suboptimal play. Your current exponential decay is a good choice.
    *   **Noisy Networks (Previously tried):** An alternative for state-dependent exploration. Could be revisited if epsilon-greedy with LSTM struggles.

**III. The Iterative Process for "Best Agent":**

1.  **Establish a Strong Baseline:** This is what you've been doing. Your non-PER, non-LSTM DQN is a baseline.
2.  **Hypothesize and Implement Incrementally:**
    *   Your hypothesis: LSTM will help with temporal patterns in Ms. Pac-Man.
    *   Implementation: Add LSTM (as you've done, with `--use_lstm`).
3.  **Rigorous Experimentation & Evaluation:**
    *   **Controlled Comparisons:** When testing LSTM, keep other factors (like PER status, learning rates if possible) consistent with your best non-LSTM baseline initially.
    *   **Multiple Runs:** Due to randomness in initialization and exploration, run each configuration several times (e.g., 3-5 runs with different seeds) to get a sense of average performance and variance.
    *   **Sufficient Training:** Ensure agents are trained for enough episodes/steps to converge or show their potential.
    *   **Key Metrics:**
        *   Average total reward per episode (over a window, e.g., last 100 episodes).
        *   Maximum reward achieved.
        *   Learning curves (reward vs. training steps/episodes).
        *   Loss curves (to check for stability and convergence).
        *   Episode lengths.
    *   **W&B Logging:** Essential for tracking all this.
4.  **Ablation Studies:**
    *   You've done this with PER (on vs. off).
    *   Now, it's non-LSTM vs. LSTM.
    *   If LSTM helps, you might then try LSTM+PER vs. LSTM without PER.
5.  **Hyperparameter Tuning:** This is often the most time-consuming part.
    *   Focus on parameters most likely to have a big impact: learning rate, LSTM hidden size, epsilon decay, target update frequency.
    *   Use a systematic approach if possible (grid search for a few, or more advanced methods if you have the infrastructure).
6.  **Qualitative Analysis: Watch Your Agent Play!**
    *   This is incredibly insightful. Save videos or render occasionally.
    *   Does it get stuck?
    *   Does it use power pellets strategically?
    *   How does it react to ghosts?
    *   Does it explore different mazes effectively?
    *   Are there obvious repetitive mistakes?
    This can give you clues about what aspects of your agent (architecture, exploration, reward shaping if you were using it) might need adjustment.
7.  **Compare to Benchmarks (if available):** Look up published results for DQN (or variants) on Ms. Pac-Man to get a sense of expected performance levels.

**Specific to the LSTM Complications We Discussed:**

*   **Training Time:** Be prepared for significantly longer training runs.
*   **"Stale" Hidden States in Replay:** This is the biggest theoretical challenge with DRQN. The zero-hidden-state for batch items is a simplification. If performance plateaus, this is an area where more advanced DRQN techniques (like storing and replaying sequences) could be explored, but they add substantial complexity.
*   **Hyperparameters:** The LSTM adds its own (hidden size) and might make the existing ones (like learning rate) behave differently.

**In conclusion, for Ms. Pac-Man with the current LSTM experiment:**

*   **Verify actions:** Double-check the 9 actions are indeed the directional ones + No-Op.
*   **Focus on Comparison:** Your primary goal now is to see if LSTM (without PER, based on your prior finding) outperforms your best non-LSTM (without PER) agent.
*   **Monitor Training:** Pay close attention to training time, stability, and learning curves.
*   **Qualitative Check:** Once trained, watch it play. Does it seem to exhibit more "intelligent" long-term behavior, or does it struggle?

## ReLU or GELU?
#### Verdict : Not worth the complutational complexity as ReLU is working completely fine until now! Let's stick with ReLU for now!
Let's consider the implications of replacing ReLU with GELU, both in the specific context of your `DQNCNN` and `DQNCNN_LSTM` and more generally.

**1. Will replacing ReLU with GELU help?**

*   **Potential Benefits (as discussed before for the FC layers):**
    *   **Smoother Activation:** GELU is smoother than ReLU, which can lead to a smoother optimization landscape and potentially more stable or slightly faster convergence.
    *   **Non-Monotonicity:** GELU can be negative for negative inputs, allowing it to model more complex functions than ReLU, which hard-zeros negative inputs. This means it weighs inputs by their magnitude rather than just gating them based on sign.
    *   **Reduced "Dying Neuron" Problem:** Since GELU has non-zero gradients for negative inputs (unlike ReLU), it might be less prone to the "dying ReLU" problem where neurons get stuck outputting zero.
    *   **Empirical Success:** GELU has shown strong performance in various architectures, especially Transformers.

*   **Context of Convolutional Layers:**
    *   ReLU has been the workhorse for CNNs for a long time and is computationally very efficient. It promotes sparsity, which can be beneficial.
    *   The benefits of GELU observed in Transformers (which rely heavily on self-attention and dense layers) might not translate with the same magnitude to the convolutional feature extraction part of a DQN.
    *   However, the theoretical advantages (smoother gradients, better handling of negative inputs) could still offer marginal improvements in the feature maps generated by the CNN.

*   **Your Current Setup:**
    *   You are already using GELU in the fully connected (FC) stream of both `DQNCNN` (after the CNN) and `DQNCNN_LSTM` (after the LSTM).
    *   The convolutional layers in both `DQNCNN` and `DQNCNN_LSTM` still use `nn.ReLU()`.

**2. Should we replace ReLU with GELU everywhere?**

This is a nuanced question:

*   **Argument for Replacing Everywhere (Consistency & Potential Small Gains):**
    *   Using GELU consistently throughout the network (both CNN and FC/LSTM-output layers) could provide its benefits more uniformly.
    *   If GELU is indeed slightly better at feature representation or gradient flow, applying it to the initial feature extractors (CNNs) might lead to marginally better features being passed to the subsequent LSTM or FC layers.

*   **Arguments Against Replacing Everywhere (or for Cautious Replacement):**
    *   **Computational Cost:** GELU is more computationally expensive than ReLU due to the calculation of the Gaussian CDF or its approximation. While this might be negligible for a few FC layers, replacing every ReLU in a deep CNN stack with GELU will increase the overall computational cost per iteration. For DQNs that require many environment steps and model updates, this can add up.
    *   **Empirical Evidence in CNNs for RL:** While GELU is dominant in Transformers, ReLU (and its variants like LeakyReLU or PReLU) is still very common and effective in CNNs for visual tasks, including RL from pixels. The performance uplift from GELU in CNNs specifically for Atari DQNs might be small and not always worth the extra computation.
    *   **Established Baselines:** Many successful DQN implementations and benchmarks (like the original Nature DQN) used ReLU in their CNNs. Sticking with ReLU in the CNN part makes it easier to compare against these established results if that's a goal.
    *   **Interaction with Initialization:** While Kaiming initialization (which you use) is designed for ReLU-like activations and generally works well with GELU too, the optimal interplay might subtly differ.
    *   **"If it ain't broke, don't fix it (aggressively)":** If your CNN with ReLU is learning reasonable features, changing it introduces another variable. It's often better to change one thing at a time or focus on areas with more known impact (like the LSTM itself, or exploration strategies).

## Training Specifications : 
* **GPU(s) :** For the DQNAgents I have done all the experiments on a single T4 GPU. 
*  Experiments include :    
    - `DQNBreakout` training without and with PER : (insert wandb run link here)
    - `DQNPac-Man` training without and with PER : (insert wandb run link here)
    - `DQNPac-Man LSTM` Final without PER : (insert wandb run link here)
    - Final agent chosen : `DQNPac-Man LSTM`