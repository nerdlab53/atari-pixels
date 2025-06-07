import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from PIL import Image
from torchvision.transforms import transforms

from vqvae_model import VQVAE
from dqn_agent import DQNCNN

# --- Configuration ---
class ModelConfig:
    def __init__(self):
        self.embedding_dim = 64
        self.latent_vocab_size = 256
        self.vqvae_checkpoint = 'model-checkpoints/ALE_MsPacman-v5_best.pth'
        self.dqn_no_per_checkpoint = 'model-weights/DQN-Ms.Pac-Man/dqn_MsPacman_NoPER_latest.pth'
        self.dqn_per_checkpoint = 'model-weights/DQN-Ms.Pac-Man/dqn_MsPacman_PER_latest.pth'
        self.dqn_input_shape = (8, 84, 84)
        self.dqn_n_actions = 9

class VisualizationConfig:
    def __init__(self):
        self.stationary_frame_path = 'data/vqvae_ms_pacman_rgb/episode_0001/00000.png'
        self.non_stationary_frame_t_path = 'data/vqvae_ms_pacman_rgb/episode_0001/00386.png'
        self.non_stationary_frame_tp1_path = 'data/vqvae_ms_pacman_rgb/episode_0001/00387.png'
        self.output_dir = 'latent_space_visualizations'

# --- Metrics ---
def psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    mu1 = F.avg_pool2d(img1, 3, 1, padding=1)
    mu2 = F.avg_pool2d(img2, 3, 1, padding=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, padding=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, padding=1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, padding=1) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def codebook_entropy(indices, codebook_size):
    hist = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    prob = hist / hist.sum()
    entropy = -(prob[prob > 0] * torch.log(prob[prob > 0])).sum().item()
    return entropy, hist.cpu().numpy()

# --- Helper Functions ---
def load_vqvae_model(device):
    """Loads the VQ-VAE model."""
    args = ModelConfig()
    model = VQVAE(
        input_channels_per_frame=3,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.latent_vocab_size
    ).to(device)
    try:
        checkpoint = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ VQ-VAE model loaded from {args.vqvae_checkpoint}")
    except FileNotFoundError:
        print(f"✗ Checkpoint not found at {args.vqvae_checkpoint}")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    return model

def load_dqn_model(checkpoint_path, input_shape, n_actions, device):
    """Loads a DQN model."""
    model = DQNCNN(
        input_shape=input_shape,
        n_actions=n_actions
    ).to(device)
    try:
        # The saved checkpoints are for the entire agent, we need to load the policy_net state_dict
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Check if the checkpoint is a state_dict itself or a dictionary containing it
        if 'policy_net_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['policy_net_state_dict'])
        elif 'model_state_dict' in checkpoint: # For compatibility
            model.load_state_dict(checkpoint['model_state_dict'])
        else: # Assume the checkpoint is the state dict
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✓ DQN model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"✗ DQN checkpoint not found at {checkpoint_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading DQN model: {e}")
        return None
    return model

def preprocess_image(image_path, device):
    """Loads and preprocesses an image for VQ-VAE."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"✗ Image not found at {image_path}")
        return None

def preprocess_image_dqn(image_path, device):
    """Loads and preprocesses an image for DQN, creating a stacked grayscale input."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((84, 84), antialias=True),
        transforms.ToTensor(),
    ])
    try:
        image = Image.open(image_path).convert('RGB') # Start with RGB for consistency
        processed_image = transform(image) # -> [1, 84, 84]
        # Stack the single frame 8 times to match DQN input shape
        stacked_frames = processed_image.repeat(8, 1, 1) # -> [8, 84, 84]
        return stacked_frames.unsqueeze(0).to(device) # -> [1, 8, 84, 84]
    except FileNotFoundError:
        print(f"✗ Image not found at {image_path}")
        return None

def analyze_and_visualize(model, frame_t, frame_tp1, case_name, output_dir, device):
    """Runs the model and generates visualizations."""
    print(f"\n--- Analyzing Case: {case_name} ---")

    with torch.no_grad():
        output = model(frame_t, frame_tp1)
        reconstructed_frame_tp1, vq_loss, _, _, perplexity, latents_e, min_encoding_indices, _ = output

    # --- Calculate Metrics ---
    psnr_val = psnr(reconstructed_frame_tp1, frame_tp1)
    ssim_val = ssim(reconstructed_frame_tp1, frame_tp1)
    entropy, hist = codebook_entropy(min_encoding_indices, model.quantizer.num_embeddings)

    print(f"  PSNR: {psnr_val:.4f}")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  Codebook Entropy: {entropy:.4f}")
    print(f"  Perplexity (from model): {perplexity.item():.4f}")
    print(f"  VQ Loss: {vq_loss.item():.6f}")

    # --- Permute for visualization ---
    # The model internals work with (H, W) -> (210, 160) but visualization needs it back
    latents_e_for_viz = latents_e.permute(0, 1, 3, 2)
    min_encoding_indices_for_viz = min_encoding_indices.view(latents_e.shape[0], latents_e.shape[2], latents_e.shape[3]).permute(0, 2, 1)


    # --- Visualize Latent Activations (Mean over embedding dim) ---
    latent_heatmap = latents_e_for_viz.mean(dim=1).squeeze().cpu().numpy()
    plt.figure(figsize=(10, 7))
    plt.imshow(latent_heatmap, cmap='viridis')
    plt.colorbar(label='Mean Activation')
    plt.title(f'Mean Latent Activations - {case_name}')
    plt.savefig(os.path.join(output_dir, f'{case_name}_latent_activation_heatmap.png'))
    plt.close()

    # --- Visualize Quantized Codebook Indices ---
    codebook_map = min_encoding_indices_for_viz.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 7))
    plt.imshow(codebook_map, cmap='tab20', vmin=0, vmax=model.quantizer.num_embeddings)
    plt.colorbar(label='Codebook Index')
    plt.title(f'Codebook Index Map - {case_name}')
    plt.savefig(os.path.join(output_dir, f'{case_name}_codebook_map.png'))
    plt.close()

    # --- Visualize Codebook Usage Histogram ---
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(hist)), hist)
    plt.title(f'Codebook Usage Histogram - {case_name}')
    plt.xlabel('Codebook Index')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f'{case_name}_codebook_histogram.png'))
    plt.close()

    # --- Visualize Frame Reconstruction ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    frame_t_vis = frame_t.squeeze().permute(1, 2, 0).cpu().numpy()
    frame_tp1_vis = frame_tp1.squeeze().permute(1, 2, 0).cpu().numpy()
    recon_vis = reconstructed_frame_tp1.squeeze().permute(1, 2, 0).cpu().numpy()

    axes[0].imshow(frame_t_vis)
    axes[0].set_title('Input Frame (t)')
    axes[0].axis('off')

    axes[1].imshow(frame_tp1_vis)
    axes[1].set_title('Target Frame (t+1)')
    axes[1].axis('off')

    axes[2].imshow(np.clip(recon_vis, 0, 1))
    axes[2].set_title('Reconstructed Frame (t+1)')
    axes[2].axis('off')

    plt.suptitle(f'Reconstruction for {case_name}\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}')
    plt.savefig(os.path.join(output_dir, f'{case_name}_reconstruction.png'))
    plt.close()

    print(f"✓ Visualizations for {case_name} saved to '{output_dir}'")


def analyze_and_visualize_dqn(model, frame, case_name, output_dir, device):
    """Runs the DQN model and visualizes its latent activations."""
    print(f"--- Analyzing DQN Case: {case_name} ---")

    with torch.no_grad():
        # The DQNCNN forward pass handles normalization, but we are calling model.conv directly.
        # So we must normalize here.
        model_input = frame.float() / 255.0
        latent_activations = model.conv(model_input)

    # --- Visualize Latent Activations (Mean over embedding dim) ---
    # The output of the conv layers is the latent space for the DQN
    latent_heatmap = latent_activations.mean(dim=1).squeeze().cpu().numpy()
    plt.figure(figsize=(8, 8))
    im = plt.imshow(latent_heatmap, cmap='viridis')
    plt.colorbar(im, label='Mean Activation')
    plt.title(f'DQN Mean Latent Activations - {case_name}')
    plt.xlabel('Latent X')
    plt.ylabel('Latent Y')
    plt.savefig(os.path.join(output_dir, f'{case_name}_dqn_latent_activation_heatmap.png'))
    plt.close()

    print(f"✓ DQN visualizations for {case_name} saved to '{output_dir}'")


def main():
    vis_config = VisualizationConfig()
    model_config = ModelConfig()
    os.makedirs(vis_config.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load VQ-VAE model
    vqvae_model = load_vqvae_model(device)
    if vqvae_model is None:
        print("Skipping VQ-VAE analysis.")
    else:
        # --- Case 1: Stationary Frame (VQ-VAE) ---
        frame_stationary = preprocess_image(vis_config.stationary_frame_path, device)
        if frame_stationary is not None:
            analyze_and_visualize(
                model=vqvae_model,
                frame_t=frame_stationary,
                frame_tp1=frame_stationary, # For stationary, t and t+1 are the same
                case_name='vqvae_stationary',
                output_dir=vis_config.output_dir,
                device=device
            )

        # --- Case 2: Non-Stationary Frame (VQ-VAE) ---
        frame_t_non_stationary = preprocess_image(vis_config.non_stationary_frame_t_path, device)
        frame_tp1_non_stationary = preprocess_image(vis_config.non_stationary_frame_tp1_path, device)

        if frame_t_non_stationary is not None and frame_tp1_non_stationary is not None:
            analyze_and_visualize(
                model=vqvae_model,
                frame_t=frame_t_non_stationary,
                frame_tp1=frame_tp1_non_stationary,
                case_name='vqvae_non_stationary',
                output_dir=vis_config.output_dir,
                device=device
            )

    print("\n" + "="*40)
    print("--- Starting DQN Analysis ---")
    print("="*40)

    # Load DQN Models
    dqn_noper_model = load_dqn_model(
        model_config.dqn_no_per_checkpoint,
        model_config.dqn_input_shape,
        model_config.dqn_n_actions,
        device
    )
    dqn_per_model = load_dqn_model(
        model_config.dqn_per_checkpoint,
        model_config.dqn_input_shape,
        model_config.dqn_n_actions,
        device
    )

    # --- Prepare frames for DQN ---
    dqn_frame_stationary = preprocess_image_dqn(vis_config.stationary_frame_path, device)
    dqn_frame_non_stationary = preprocess_image_dqn(vis_config.non_stationary_frame_t_path, device)

    # --- Analyze DQN Models ---
    if dqn_noper_model is not None:
        if dqn_frame_stationary is not None:
            analyze_and_visualize_dqn(dqn_noper_model, dqn_frame_stationary, 'dqn_noper_stationary', vis_config.output_dir, device)
        if dqn_frame_non_stationary is not None:
            analyze_and_visualize_dqn(dqn_noper_model, dqn_frame_non_stationary, 'dqn_noper_non_stationary', vis_config.output_dir, device)

    if dqn_per_model is not None:
        if dqn_frame_stationary is not None:
            analyze_and_visualize_dqn(dqn_per_model, dqn_frame_stationary, 'dqn_per_stationary', vis_config.output_dir, device)
        if dqn_frame_non_stationary is not None:
            analyze_and_visualize_dqn(dqn_per_model, dqn_frame_non_stationary, 'dqn_per_non_stationary', vis_config.output_dir, device)


if __name__ == '__main__':
    main() 