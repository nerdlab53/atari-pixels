"""
Generate a dataset of (action, latent_code) pairs for training the Action Mapping Model.

This script uses a random agent to explore the game and a trained VQ-VAE to
encode the resulting frame transitions into discrete latent codes.

The output is a JSON file containing a list of dictionaries, where each dictionary
represents a single transition: {'action': int, 'latent_code': [int, ...]}
"""

import os
import numpy as np
import torch
import gymnasium as gym
import argparse
from tqdm import tqdm
import json
from torchvision import transforms as T

from vqvae_model import VQVAE # Correct model import

def load_vqvae_model(model_path, device):
    """Loads the VQVAE model with its specific checkpoint structure."""
    # These are the arguments the model was trained with, based on vqvae_model.py
    # and the errors we've seen.
    args = argparse.Namespace(
        input_channels_per_frame=3,
        embedding_dim=64, # Based on previous error analysis
        num_embeddings=256,
        commitment_cost=0.25,
        dropout_p=0.1
    )
    model = VQVAE(
        input_channels_per_frame=args.input_channels_per_frame,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        dropout_p=args.dropout_p
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # The checkpoint is a dictionary, and the weights are under the 'model_state_dict' key
    state_dict = checkpoint['model_state_dict']
    
    # Handle compiled models by removing the prefix
    unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped_state_dict)
    
    return model

def get_env(env_name, seed):
    """Creates and wraps the Atari environment."""
    env = gym.make(env_name, render_mode='rgb_array') # Need RGB frames
    env.action_space.seed(seed)
    return env
    
def generate_action_latent_pairs(
    model_path: str,
    env_name: str,
    output_path: str,
    num_pairs: int,
    seed: int = 42,
):
    """Generates (action, latent_code) pairs."""
    # --- Device Setup ---
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Environment Setup ---
    env = get_env(env_name, seed)
    n_actions = env.action_space.n
    print(f"Environment '{env_name}' created with {n_actions} actions.")
    
    # --- Load VQ-VAE Model ---
    print(f"Loading VQ-VAE model from {model_path}...")
    vqvae = load_vqvae_model(model_path, device)
    vqvae.eval()
    # vqvae = torch.compile(vqvae) # Disabled due to a bug in the Metal backend for torch.compile
    print("VQ-VAE model loaded.")
    
    # --- Data Collection Loop ---
    collected_pairs = []
    
    # Transformation to resize to (210, 160) and convert to tensor
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((210, 160)),
        T.ToTensor() # This also scales to [0, 1]
    ])

    frame_t_np, _ = env.reset(seed=seed)
    
    pbar = tqdm(range(num_pairs), desc="Collecting action-latent pairs")
    for _ in pbar:
        action = env.action_space.sample()
        frame_tp1_np, _, terminated, truncated, _ = env.step(action)
        
        frame_t_tensor = transform(frame_t_np).unsqueeze(0).to(device)
        frame_tp1_tensor = transform(frame_tp1_np).unsqueeze(0).to(device)

        # Use VQ-VAE to get latent indices for the transition
        with torch.no_grad():
            # The model returns a tuple, the indices are the 7th element (index 6)
            _, _, _, _, _, _, min_encoding_indices, _ = vqvae(frame_t_tensor, frame_tp1_tensor)
        
        latent_code = min_encoding_indices.view(-1).cpu().numpy().tolist()

        collected_pairs.append({'action': int(action), 'latent_code': latent_code})
        
        pbar.set_postfix({"action": action, "latent_len": len(latent_code)})

        if terminated or truncated:
            frame_t_np, _ = env.reset()
        else:
            frame_t_np = frame_tp1_np

    env.close()

    # --- Save Dataset as JSON ---
    print(f"\nFinished data collection. Saving {len(collected_pairs)} pairs...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(collected_pairs, f, indent=2)
        
    print(f"Dataset saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate (action, latent_code) pairs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained VQ-VAE .pth model file.")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5", help="Name of the Atari environment.")
    parser.add_argument("--output_path", type=str, default="data/actions/action_latent_pairs.json", help="Path to save the output .json dataset.")
    parser.add_argument("--num_pairs", type=int, default=50000, help="Total number of pairs to collect.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for the environment.")
    
    args = parser.parse_args()

    generate_action_latent_pairs(
        model_path=args.model_path,
        env_name=args.env_name,
        output_path=args.output_path,
        num_pairs=args.num_pairs,
        seed=args.seed
    ) 