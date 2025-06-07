# Part 4: Value Model for Latent State Evaluation in Ms. Pac-Man

## Overview
This part focuses on building a value model that can predict the expected future reward from a given latent state representation. Using the latent codes learned by the VQ-VAE from Part 2, we train a simple MLP to estimate the value of game states, which could be useful for planning or reinforcement learning applications.

## Main Focus:

- My main focus through this experiment was to:
    - Build a value function that operates on VQ-VAE latent representations
    - Train the model to predict expected rewards from (latent_state, reward) pairs
    - Evaluate whether the compressed latent space retains enough information for value estimation
    - Create a component that could potentially be used for model-based RL

- Build a value estimation pipeline:
    - Extract latent state representations using the trained VQ-VAE
    - Collect reward data from the DQN agent gameplay episodes
    - Train an MLP to map latent states to expected future rewards

## Implementation Details

### Data Preparation
1. Create (latent_state, reward) pairs from the gameplay data:
    - Use the trained VQ-VAE to encode game frames into 5×7 grids of discrete latent codes
    - Extract corresponding reward information from the DQN agent episodes
    - Flatten the 5×7 latent grid into sequences of 35 discrete tokens
    - Create train (90%) and validation (10%) splits

2. Data format:
    - Input: Sequences of 35 latent codes (integers in [0, 255])
    - Target: Single reward value (float)
    - No preprocessing needed since working with discrete indices

### Value Model Architecture

The implemented architecture is a straightforward MLP that processes embedded latent states:

**Final Implemented Architecture:**
1. **Embedding Layer:**
    - Maps each of the 256 possible VQ-VAE codes to 64-dimensional vectors
    - Input: (batch_size, 35) → Output: (batch_size, 35, 64)

2. **Flattening:**
    - Flattens embedded sequence: (batch_size, 35, 64) → (batch_size, 2240)

3. **MLP Head:**
    - Linear layer: 2240 → 512 with ReLU and Dropout (0.2)
    - Linear layer: 512 → 256 with ReLU and Dropout (0.2)  
    - Linear layer: 256 → 1 (scalar value output)

### Training Process
1. **Optimizer:** Adam with learning rate 1e-4
2. **Loss Function:** Mean Squared Error (MSE) between predicted and actual rewards
3. **Training Details:**
    - Batch size: 256
    - Training epochs: 50
    - Additional metric: Mean Absolute Error (MAE) for monitoring
4. **Logging:** WandB tracking for loss curves and evaluation metrics, see the training here [WandB Value Model](https://wandb.ai/retr0sushi-04/atari-value-model-ms-pacman-v3?nw=nwuserretr0sushi04)

## Experiment Results and Analysis

### Training Results
The value model training showed clear learning patterns with some overfitting:

**Training Performance:**
- **Initial Training MSE:** ~0.9 (epoch 1)
- **Final Training MSE:** ~0.16-0.18 (epoch 50)
- **Training Improvement:** 80-82% reduction in MSE loss

**Validation Performance:**
- **Initial Validation MSE:** ~0.75 (epoch 1)
- **Final Validation MSE:** ~0.34-0.35 (epoch 50)
- **Validation Improvement:** 53-55% reduction in MSE loss

**Overfitting Analysis:**
- Clear divergence between training and validation curves after epoch 10
- Training loss continued decreasing while validation loss plateaued around 0.34
- Gap between training (0.16) and validation (0.34) indicates moderate overfitting
- The model memorized training patterns rather than generalizing fully

### Training Dynamics
- **Fast Initial Learning:** Both curves dropped rapidly in the first 10 epochs
- **Training Convergence:** Training loss smoothly decreased throughout 50 epochs
- **Validation Plateau:** Validation loss stabilized around epoch 15-20
- **Stable Training:** No signs of instability or loss spikes during training

### Model Behavior
- **Strengths:** 
  - Successfully learned to predict rewards from latent representations
  - Stable training with clear convergence
  - Significant improvement from baseline on both training and validation
- **Limitations:** 
  - Moderate overfitting with 2x gap between train/val performance
  - Validation performance plateaued suggesting limited generalization
- **Latent Space Utility:** VQ-VAE codes clearly retained reward-relevant information

### Key Observations
- The latent representations contained sufficient signal for value estimation
- Simple MLP architecture learned the task but had capacity for overfitting
- Results suggest the approach is viable but would benefit from regularization

## Experimentation Notes and Timeline:

- This phase was relatively straightforward compared to the previous VQ-VAE and world model training.

- **Implementation Simplicity:** The MLP approach was simple and quick to implement

- **Data Pipeline:** Most time was spent setting up the data pipeline to extract (latent_state, reward) pairs

- **Training Speed:** The value model trained quickly due to its simple architecture

- **Total Time:** Spent about 1.5 hours total including data preparation and training

- **Main Challenge:** Ensuring the data pipeline correctly aligned latent states with corresponding rewards

## My Learnings, Observations, Final Takes and Improvements Regarding This Phase 4 of Ms.PacMan:

- **Latent Space Success:** The VQ-VAE latent representations proved genuinely useful for value estimation, with training MSE dropping from 0.9 to 0.16

- **Overfitting Reality:** The 2x gap between training (0.16) and validation (0.34) MSE showed clear overfitting, highlighting the need for better regularization

- **Simple Baseline Works:** A basic MLP was sufficient to demonstrate the concept and achieve reasonable performance

- **Data Quality Matters:** The training curves showed stable learning, suggesting the (state, reward) pairs from DQN were of good quality

- **Practical Insights:**
  - Early stopping around epoch 15-20 would likely improve generalization
  - More aggressive dropout or L2 regularization could help reduce overfitting
  - The approach validates that VQ-VAE latent spaces retain task-relevant information

- **What Could Be Improved:**
  - Early stopping based on validation loss
  - More sophisticated regularization techniques (L2, higher dropout)
  - Cross-validation to better assess generalization
  - Larger validation set for more reliable overfitting detection

- This phase successfully demonstrated that latent representations can be used for downstream value estimation, with clear quantitative evidence of learning, though the overfitting suggests room for architectural improvements. 