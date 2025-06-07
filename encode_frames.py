"""
Encodes raw game frames into VQ-VAE latent codes and pairs them with actions.

This script loads a trained VQ-VAE model and uses its encoder to transform
a directory of game frames (e.g., .png files) into sequences of discrete
latent codes. It pairs these codes with corresponding actions from a
JSON file and saves the output, which can then be used to create a
dataset for training a world model.
"""
import torch
import os
import json
import argparse
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Assumption: You have a vq_vae.py file with your VQVAE model definition.
# If your model class is named differently or is in another file,
# you will need to change the import statement below.
from vqvae_model import VQVAE

def encode_frames(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load VQ-VAE Model ---
    print(f"Loading VQ-VAE model from {args.vqvae_checkpoint}...")
    # Correctly initialize the VQVAE model based on its definition
    model = VQVAE(
        input_channels_per_frame=3,
        embedding_dim=64,
        num_embeddings=256,
        commitment_cost=0.25,
        dropout_p=0.1
    ).to(device)
    model.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    print("VQ-VAE model loaded.")

    # --- Load Action Data ---
    # This assumes your actions file is a JSON file containing a list of actions,
    # where the index of the list corresponds to the frame number.
    print(f"Loading actions from {args.actions_path}...")
    with open(args.actions_path, 'r') as f:
        actions = json.load(f)
    print(f"Loaded {len(actions)} actions.")

    # --- Image Transformations ---
    # Define the same transformations that were used when training the VQ-VAE.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # The reference model does not use normalization, but we keep it commented
        # in case your training procedure was different.
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- Process Frames ---
    action_latent_pairs = []
    frame_files = sorted([f for f in os.listdir(args.frames_dir) if f.endswith(('.png', '.jpg'))])
    
    # We need pairs of frames (t, t+1), so we iterate up to the second to last frame.
    print(f"Encoding {len(frame_files) - 1} frame pairs from {args.frames_dir}...")
    for i in tqdm(range(len(frame_files) - 1), desc="Encoding frame pairs"):
        if i >= len(actions):
            print(f"Warning: Ran out of actions at index {i}. Stopping.")
            break

        # Load frame t and frame t+1
        frame_t_path = os.path.join(args.frames_dir, frame_files[i])
        frame_tp1_path = os.path.join(args.frames_dir, frame_files[i+1])
        
        image_t = Image.open(frame_t_path).convert("RGB")
        image_tp1 = Image.open(frame_tp1_path).convert("RGB")
        
        tensor_t = transform(image_t).unsqueeze(0).to(device)
        tensor_tp1 = transform(image_tp1).unsqueeze(0).to(device)

        with torch.no_grad():
            # The VQ-VAE model expects permuted frames (C, W, H) instead of (C, H, W)
            frame_t_permuted = tensor_t.permute(0, 1, 3, 2)
            frame_tp1_permuted = tensor_tp1.permute(0, 1, 3, 2)
            
            # Concatenate frames to find the latent "action"
            x = torch.cat([frame_t_permuted, frame_tp1_permuted], dim=1)
            
            # 1. Pass through the encoder
            latents_e = model.encoder(x)
            
            # 2. Pass through the quantizer to get the discrete codes
            _, _, _, _, _, min_encoding_indices = model.quantizer(latents_e)
            
            # Reshape indices to a flat list
            latent_codes = min_encoding_indices.view(1, -1).squeeze(0).cpu().tolist()

        # We pair the latent code representing the transition from t to t+1
        # with the action taken at time t.
        action_latent_pairs.append({
            "action": actions[i],
            "latent_code": latent_codes
        })

    # --- Save Output ---
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(action_latent_pairs)} action-latent pairs to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(action_latent_pairs, f)

    print("Encoding complete.")


def main():
    parser = argparse.ArgumentParser(description="Encode game frames using a trained VQ-VAE.")
    # --- IMPORTANT: You must provide these paths ---
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing the raw game frames (e.g., PNG files).")
    parser.add_argument("--actions_path", type=str, required=True, help="Path to the JSON file containing the list of actions for each frame.")
    parser.add_argument("--vqvae_checkpoint", type=str, required=True, help="Path to the trained VQ-VAE model checkpoint (.pt or .pth file).")
    
    # --- Output path ---
    parser.add_argument("--output_path", type=str, default="data/actions/action_latent_pairs.json", help="Path to save the output action-latent pairs JSON file.")
    
    args = parser.parse_args()
    encode_frames(args)

if __name__ == "__main__":
    main() 