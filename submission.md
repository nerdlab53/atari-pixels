# Neural Atari Research Submission: Ms. Pac-Man Multi-Model Analysis

## Project Development Process

This project was completed in five phases, each building upon the previous work:

### [Part 1: DQN Training with Prioritized Experience Replay](part1_mspacman.md)
- Implementation of DQN agent for Ms. Pac-Man
- Comparison between standard experience replay and Prioritized Experience Replay (PER)
- Performance analysis and attention pattern visualization

### [Part 2: VQ-VAE for Frame Representation Learning](part2_mspacman.md)
- Development of Vector Quantized Variational Autoencoder for game frame encoding
- Exploration of different architectures and hyperparameters
- Analysis of learned latent representations and codebook utilization

### [Part 3: Next-Frame Prediction with Latent Actions](part3_mspacman.md)
- Integration of VQ-VAE with action-conditioned frame prediction
- Implementation of attention mechanisms for multi-modal learning
- Evaluation of temporal consistency and prediction quality

### [Part 4: Action Space Analysis and Mapping](part4_mspacman.md)
- Statistical analysis of learned latent codes and their relationship to game actions
- Development of action classification models
- Investigation of representation interpretability

### [Part 5: Multi-Model Representation Comparison](part5_mspacman.md)
- Comparative analysis of representations learned by different models
- Attention pattern analysis across DQN variants
- Integration of findings and cross-model insights

### Finally the answers to the questions asked are present in this [file](responses.md)

### Thank you for providing me the opportunity to work on such a cool project!