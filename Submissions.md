# Project 3 Submission: Extending Neural Atari

This document details the work done to extend the Neural Atari project, focusing on analyzing and comparing the latent representations of different neural network models trained on `Ms. Pac-Man`.

**Submission Link:** This GitHub repository serves as the submission. It contains all the code used for training the agents, visualizing their latent spaces, and the resulting visualizations.

### Why did you pick this project?

I chose this project due to a deep interest in understanding the internal workings of neural networks, especially in the context of reinforcement learning. While achieving high scores in games is a common benchmark, the question of *how* an agent perceives its environment to make decisions is arguably more fascinating and fundamental.

The "Neural Atari" project provided an excellent starting point. The idea of compressing game states into a latent space with a VQ-VAE and then analyzing it is a direct path to exploring concepts like world models and state representation. This project offered the creative freedom to "do something interesting," which I interpreted as a call to move beyond surface-level performance metrics and into the realm of model interpretability. By comparing a generative model (VQ-VAE) with a decision-making model (DQN), and further comparing DQNs with different training dynamics (with and without Prioritized Experience Replay), I aimed to uncover how an agent's objectives and experiences shape its internal "worldview."

### If you had more compute/time, what would you have done?

With additional resources, I would have pursued several exciting extensions:

*   **Train a Full World Model:** The VQ-VAE serves as the perceptual backbone. The logical next step is to train a predictive model (e.g., an LSTM or Transformer) on top of the VQ-VAE's latent space. This model would learn to predict the next latent state (`z_{t+1}`) given the current state and action (`z_t`, `a_t`), effectively creating a "world model." This would allow the agent to simulate future outcomes and plan ahead, forming the basis for model-based reinforcement learning.
*   **Train an Agent Entirely in Latent Space:** Instead of using pixels, an agent could be trained to play the game using the compressed latent codes from the VQ-VAE as its state input. This could lead to vastly more sample-efficient learning, as the agent would operate on a much smaller, more abstract, and potentially more meaningful state representation.
*   **Broader Comparative Analysis:** I would apply this analysis pipeline to a wider variety of Atari games (e.g., `Space Invaders`, `Pitfall!`, `Seaquest`) to conduct a comparative study. This would reveal how different game mechanics—such as static vs. scrolling screens, or the need for long-term planning vs. quick reflexes—lead to fundamentally different learned representations.
*   **Investigate Temporal Dynamics:** Following the optional prompt, I would train the models on games with an explicit time component, like a visible timer or a day/night cycle. Analyzing the generated frames from a world model would reveal how the network learns to represent the passage of time without an external clock. Does it learn to "tick" the timer correctly, or does time behave in strange, non-linear ways in the network's imagination?

### What did you learn in the project?

This project was a rich learning experience, both conceptually and practically.

1.  **Representations are Task-Dependent:** The most significant takeaway was seeing how a model's objective function dictates its learned representation. The VQ-VAE, trained on reconstruction loss, learned to encode features necessary to redraw the entire scene faithfully. In contrast, the DQN, trained to maximize future rewards, learned to focus its activations on features critical for gameplay—the positions of Ms. Pac-Man, the ghosts, and the pellets. The world looks different to an artist than it does to a predator.
2.  **The Power of Vector Quantization:** I was impressed by how effectively the VQ-VAE's discrete codebook could compress complex game screens into a small set of latent codes while preserving high visual fidelity upon reconstruction. This highlights the power of learning a compact, expressive "visual vocabulary" for complex data.
3.  **Practical Model Analysis:** I gained hands-on experience in loading and dissecting pretrained PyTorch models, modifying them to extract intermediate representations, and developing a workflow for generating and comparing visualizations that yield scientific insights.

### What surprised you the most?

*   **The Locality of DQN Features:** It was striking to see how cleanly the DQN's convolutional feature maps localized on key game elements. The network wasn't just vaguely processing the screen; it was demonstrably "looking" at specific, important objects. This provided a very direct and intuitive confirmation of what we hope these agents are doing.
*   **Subtlety in Representation Differences:** While I expected to see a difference between the PER and non-PER DQN agents, the visual difference in their mean latent activations was not as stark as anticipated. This was surprising and suggests that both agents converge on learning a similar set of core visual features necessary for playing Ms. Pac-Man. The key difference imparted by PER might lie not in *what* the agent sees, but in how it *values* the states derived from those features, which is a more subtle quality not fully captured by our current visualization. This pushes the inquiry to a deeper level.

### If you had to write a paper on the project, what else needs to be done?

While our visual analysis is insightful, a formal research paper would require more rigorous, quantitative, and causal experiments.

1.  **Quantitative Representation Analysis:**
    *   **Probing Tasks:** Freeze the learned encoders and train small, linear classifiers ("probes") on the latent representations to predict game-specific properties (e.g., "What quadrant is Ms. Pac-Man in?", "How many ghosts are on screen?"). The performance of these probes would quantitatively measure what information is explicitly encoded in the representations.
    *   **Representation Similarity Metrics:** Use metrics like Centered Kernel Alignment (CKA) to numerically compare the similarity of representations across different layers and between different models (VQ-VAE vs. DQN, PER vs. No-PER). This would move beyond visual inspection to a hard number.
2.  **Causal Interventions:**
    *   **Latent Space Manipulation:** Systematically traverse the VQ-VAE's latent space, decode the manipulated vectors, and observe the changes in the generated image. This would help establish a causal link between specific latent codes and the visual features they represent (e.g., a ghost's color or position).
    *   **Behavioral Analysis:** Correlate representation similarity with policy similarity. Do states with similar latent representations elicit similar action distributions from the DQN agent? This would connect the agent's "perception" to its "behavior."
3.  **Controlled Experiments:** Design experiments on simplified or modified versions of the game. For example, train an agent in an environment with no ghosts and compare its learned representation to an agent trained in the standard environment. This would help isolate which features are learned in response to specific environmental pressures.
4.  **Literature Review and Positioning:** A comprehensive literature review on representation learning in RL, world models, and model interpretability would be essential to properly frame the research, contextualize the findings, and clearly articulate the paper's unique contributions. 