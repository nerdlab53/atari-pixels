# Neural Atari: Ms. Pac-Man Extension

This project extends the original [atari-pixels](https://github.com/paraschopra/atari-pixels) framework to Ms. Pac-Man, exploring how different neural architectures learn representations of complex game environments through a multi-phase approach combining reinforcement learning, generative modeling, and representation analysis.

---

## Project Overview

### Phase-by-Phase Development

**[Part 1: DQN Training with Prioritized Experience Replay](part1_mspacman.md)**

**[Part 2: VQ-VAE for Frame Representation Learning](part2_mspacman.md)**  

**[Part 3: Next-Frame Prediction with Latent Actions](part3_mspacman.md)**

**[Part 4: Action Space Analysis and Mapping](part4_mspacman.md)**

**[Part 5: Multi-Model Representation Comparison](part5_mspacman.md)**

---

## Complete Documentation

- **[Research Submission](submission.md)**: Complete overview 
- **[Research Questions & Analysis](responses.md)**: Answers to the questions required for submission of the project

---

## Quick Start

### Requirements
```bash
pip install torch torchvision gymnasium numpy tqdm wandb matplotlib opencv-python pillow
```

### Training Pipeline
1. Train DQN agents (Part 1)
2. Generate gameplay data 
3. Train VQ-VAE (Part 2)
4. Train next-frame prediction models (Part 3)
5. Analyze representations (Parts 4-5)

See individual part documents for detailed instructions, code examples and the answers to the required questions by Lossfunk.