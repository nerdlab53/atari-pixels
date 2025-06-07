# Part 3: Latent Space World Model for Ms. Pac-Man

## Overview
This part focuses on building a recurrent world model that operates entirely in the latent space learned by the VQ-VAE from Part 2. Rather than predicting actual next frames, this model predicts the next latent codes given the current latent state and an action. The goal is to learn the "physics" of the game at the latent level, enabling efficient planning and rollouts without expensive pixel-level generation.

## Main Focus:

- My main focus through this experiment was to build a latent-space world model that could:
    - Take current latent codes (5×7 grid from VQ-VAE) and a game action as input
    - Predict the next set of latent codes that would result from taking that action
    - Learn temporal dynamics and spatial relationships within the compressed latent representation
    - Enable fast rollouts for planning or model-based RL applications

- Build a recurrent dynamics model:
    - Use the trained VQ-VAE from Part 2 to extract latent representations for the entire dataset
    - Train a GRU-based model to predict latent transitions: (latent_t, action) → latent_t+1

## Implementation Details

### Data Preparation
1. Use the trained VQ-VAE from Part 2 to extract latent action indices for all frame pairs in the dataset:
    - Process the entire episode dataset to create triplets: (latent_codes_t, action, latent_codes_t+1)
    - The latent codes come from the VQ-VAE's quantizer output (5×7 grid of discrete indices)
    - Each latent code is an integer in [0, 255] representing one of 256 codebook entries
    - Create train (80%) and validation (20%) splits at the episode level
2. Preprocessing transformations:
    - Flatten the 5×7 latent grid into a sequence of 35 discrete tokens
    - Actions are already discrete (9 possible Ms. Pac-Man actions)
    - No normalization needed since we're working with discrete indices

### Latent Space World Model Architecture

The implemented architecture uses a recurrent approach to model the temporal dynamics of the latent space:

**Final Implemented Architecture:**
1. **Embedding Layers:**
    - **Latent Embedding:** Maps each of the 256 possible VQ-VAE codes to 64-dimensional vectors
    - **Action Embedding:** Maps each of the 9 game actions to 64-dimensional vectors

2. **Input Fusion:**
    - Current latent codes: (batch_size, 35) → embedded to (batch_size, 35, 64)
    - Actions: (batch_size, 1) → embedded and repeated to (batch_size, 35, 64)
    - Fusion by element-wise addition to condition each spatial position on the action

3. **Recurrent Core:**
    - **GRU Architecture:** 2-layer GRU with 512 hidden units per layer
    - Processes the sequence of 35 latent positions to model spatial relationships
    - Output: (batch_size, 35, 512) hidden representations

4. **Output Head:**
    - Linear layer mapping GRU output to logits over 256 possible latent codes
    - Predicts the next latent code for each of the 35 positions independently

### Training Process
1. **Optimizer:** Adam with learning rate 1e-3
2. **Loss Function:** Cross-entropy loss between predicted and actual next latent codes
3. **Training Details:**
    - Batch size: 32
    - Learning rate scheduling: StepLR with decay every 30 epochs
4. **Logging:** WandB tracking for loss curves and prediction accuracy: [World Model Training](https://wandb.ai/retr0sushi-04/atari-world-model-ms-pac-man?nw=nwuserretr0sushi04)

## Experiment Results and Analysis

### Training Results
The GRU-based world model showed steady learning progress:
- **Training Performance:** Reached 87% per-position accuracy after convergence
- **Validation Performance:** Achieved 78% per-position accuracy, showing some overfitting
- **Training Time:** Converged within about 100 steps (roughly 45 minutes - 1 hour of training)
- **Loss Curves:** Both training and validation loss decreased steadily, with training loss reaching ~0.4 and validation loss plateauing around 0.8

### Model Behavior
- **Strengths:** The model learned to predict stationary elements well and captured basic movement patterns
- **Limitations:** Struggled with complex interactions and rare game states
- **Multi-step Prediction:** Performance degraded quickly beyond 2-3 steps ahead due to error accumulation

### Key Observations
- The accuracy gap between training (87%) and validation (78%) indicated some overfitting
- The model was most successful at predicting background maze elements and simple object movements
- Dynamic objects like Ms. Pac-Man and ghosts were harder to predict accurately

## Experimentation Notes and Timeline:

- Working in latent space made this phase much more manageable than pixel-level prediction.

- **Training Efficiency:** The discrete tokens (35 codes vs 210×160×3 pixels) made training much faster

- **Model Simplicity:** The GRU approach was straightforward compared to complex attention mechanisms

- **Debugging:** When predictions were wrong, I could directly see which latent codes were incorrect

- **Main Challenge:** Multi-step predictions degraded quickly due to error accumulation, also I was able to see a bit of overfitting at the end due to the nature of the losses.

## My Learnings, Observations, Final Takes and Improvements Regarding This Phase 3 of Ms.PacMan:

- Working in VQ-VAE's latent space was a very different approach than pixel-level prediction, but it is one that I wanted to take to see if I can still take this different approach and make the whole thing work.

- Classification on discrete codes was easier than continuous regression

- Treating the 5×7 grid as a sequence worked well for spatial relationships

- Long-horizon prediction remained challenging due to error accumulation

- While I think the GRU approach was effective without needing complex attention mechanisms, I think a very interesting experiment would be definitely to incorporate attention into this, I wasn't able to do this due to time and compute constraints due to the deadline, but this is something I would definitely take up after submitting this.

- **What Could Be Improved:**
  - Better handling of rare game states, I don't think a single GRU suffices for this, a better mechanism could help to understand the temporal dynamics of the game.
  - Longer training sequences for better temporal modeling as model can of course work better with even longer sequences.

- Through this phase I tried to figure out if latent space world models can be both efficient and reasonably effective for learning game dynamics, I was not entirely convinced to approach with this model, I still though it did a decent enough job to move further with it at the time. 