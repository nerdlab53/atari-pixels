"""Generate a dataset for training a latent action predictive model (world model).

This script uses a trained DQN agent to play the game and a trained VQ-VAE to
encode the transitions between frames into discrete latent codes.

The output is an .npz file containing tuples of:
(agent_observation, action_taken, resulting_latent_indices)
"""

import os
import numpy as np
import torch
import gymnasium as gym
import argparse
from tqdm import tqdm

from dqn_agent import DQNAgent
from train_dqn import AtariEnv
from vqvae_model import VQVAE

def generate_latent_dataset(
    dqn_model_path: str,
    vqvae_model_path: str,
    env_name: str,
    output_path: str,
    num_steps: int,
    seed: int = 42,
):
    """
    Generates a dataset for the world model.

    Args:
        dqn_model_path: Path to the trained DQN agent model.
        vqvae_model_path: Path to the trained VQ-VAE model.
        env_name: Name of the Atari environment.
        output_path: Path to save the output .npz file.
        num_steps: Total number of steps/transitions to collect.
        seed: Environment seed.
    """
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Environment Setup ---
    # The agent was trained on stacked frames, but the VQ-VAE was trained on single raw frames.
    # We need both. `AtariEnv` provides the stacked frames for the agent, and `env.render()`
    # provides the raw frames for the VQ-VAE.
    env = AtariEnv(game_id=env_name, num_stack=4, seed=seed, render_mode='rgb_array')
    n_actions = env.action_space.n
    agent_state_shape = env.observation_space.shape
    
    # --- Load DQN Agent ---
    print(f"Loading DQN agent from {dqn_model_path}...")
    agent = DQNAgent(n_actions=n_actions, state_shape=agent_state_shape, device=device)
    agent.policy_net.load_state_dict(torch.load(dqn_model_path, map_location=device))
    agent.policy_net.eval()
    print("DQN agent loaded.")

    # --- Load VQ-VAE Model ---
    print(f"Loading VQ-VAE model from {vqvae_model_path}...")
    # We need to know the parameters the VQ-VAE was trained with.
    # For now, assuming defaults. A more robust way is to save args in the checkpoint.
    vqvae_checkpoint = torch.load(vqvae_model_path, map_location=device)
    vqvae_args = vqvae_checkpoint.get('args')
    
    if vqvae_args:
        print("Found VQ-VAE args in checkpoint.")
        vqvae = VQVAE(
            input_channels_per_frame=vqvae_args.input_channels_per_frame,
            embedding_dim=vqvae_args.embedding_dim,
            num_embeddings=vqvae_args.num_embeddings,
            commitment_cost=vqvae_args.commitment_cost,
            dropout_p=getattr(vqvae_args, 'dropout_p', 0.1) # Safely get dropout
        ).to(device)
    else:
        print("Warning: VQ-VAE args not in checkpoint, using defaults. This may fail if defaults are wrong.")
        vqvae = VQVAE(embedding_dim=64, num_embeddings=256).to(device) # Example defaults

    # Handle both compiled and non-compiled state dicts
    state_dict = vqvae_checkpoint['model_state_dict']
    # If model was compiled, keys might start with '_orig_mod.'. We remove this prefix.
    unwrapped_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    vqvae.load_state_dict(unwrapped_state_dict)
    
    vqvae.eval()
    print("VQ-VAE model loaded.")
    
    # --- Data Collection Loop ---
    collected_observations = []
    collected_actions = []
    collected_latents = []

    obs, _ = env.reset(seed=seed) # Agent observation (stacked frames)
    raw_frame_t = env.render()   # Raw frame for VQ-VAE

    pbar = tqdm(range(num_steps), desc="Collecting transitions")
    for step in pbar:
        # Agent selects action based on its observation
        action = agent.select_action(torch.from_numpy(obs).float().unsqueeze(0).to(device), epsilon=0.01)

        # Environment steps
        next_obs, reward, terminated, truncated, info = env.step(action)
        raw_frame_tp1 = env.render() # Get the next raw frame

        # Convert raw frames to tensors for VQ-VAE
        # Shape: (1, C, H, W), normalized to [0,1]
        frame_t_tensor = torch.from_numpy(raw_frame_t.transpose(2,0,1)).float().unsqueeze(0).to(device) / 255.0
        frame_tp1_tensor = torch.from_numpy(raw_frame_tp1.transpose(2,0,1)).float().unsqueeze(0).to(device) / 255.0

        # Use VQ-VAE to get latent indices for the transition
        with torch.no_grad():
            # The VQVAE model now handles the permutation internally.
            _, _, _, _, _, _, min_encoding_indices, _ = vqvae(frame_t_tensor, frame_tp1_tensor)
        
        # min_encoding_indices is (B, H_latent, W_latent), we flatten it
        latent_code = min_encoding_indices.view(-1).cpu().numpy()

        # Store the collected data
        collected_observations.append(obs)
        collected_actions.append(action)
        collected_latents.append(latent_code)

        if terminated or truncated:
            obs, _ = env.reset()
            raw_frame_t = env.render()
        else:
            obs = next_obs
            raw_frame_t = raw_frame_tp1

    env.close()

    # --- Save Dataset ---
    print(f"\nFinished data collection. Saving {len(collected_actions)} transitions...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.array(collected_observations, dtype=np.uint8), # Save as uint8 to save space
        actions=np.array(collected_actions, dtype=np.int8),
        latents=np.array(collected_latents, dtype=np.int16)
    )
    print(f"Dataset saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset for latent action prediction.")
    parser.add_argument("--dqn_model_path", type=str, required=True, help="Path to the trained DQN .pth model file.")
    parser.add_argument("--vqvae_model_path", type=str, required=True, help="Path to the trained VQ-VAE .pth model file.")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5", help="Name of the Atari environment.")
    parser.add_argument("--output_path", type=str, default="data/world_model_dataset.npz", help="Path to save the output .npz dataset.")
    parser.add_argument("--num_steps", type=int, default=50000, help="Total number of transitions to collect.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for the environment.")
    
    args = parser.parse_args()

    generate_latent_dataset(
        dqn_model_path=args.dqn_model_path,
        vqvae_model_path=args.vqvae_model_path,
        env_name=args.env_name,
        output_path=args.output_path,
        num_steps=args.num_steps,
        seed=args.seed
    ) 