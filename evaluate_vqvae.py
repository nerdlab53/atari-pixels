import torch
import torch.nn.functional as F
import numpy as np
import os
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import glob

from vqvae_model import VQVAE

# --- Configuration ---
class EvaluationConfig:
    def __init__(self):
        self.embedding_dim = 64
        self.latent_vocab_size = 256
        self.model_checkpoint = 'model-checkpoints/ALE_MsPacman-v5_best.pth'
        self.data_dir = 'data/vqvae_ms_pacman_rgb'  # Directory containing episode folders
        self.batch_size = 8
        self.num_workers = 4

# --- Metrics ---
def psnr(pred, target):
    """Calculate PSNR between predicted and target images."""
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))

def ssim(img1, img2):
    """Calculate SSIM between two images."""
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

# --- Dataset ---
class FramePairDataset(Dataset):
    """Dataset for loading consecutive frame pairs from episode directories."""
    
    def __init__(self, data_dir, transform=None, max_episodes=None):
        self.data_dir = data_dir
        self.transform = transform
        self.frame_pairs = []
        
        # Find all episode directories
        episode_dirs = glob.glob(os.path.join(data_dir, 'episode_*'))
        episode_dirs.sort()
        
        if max_episodes:
            episode_dirs = episode_dirs[:max_episodes]
        
        print(f"Found {len(episode_dirs)} episode directories")
        
        # Collect all valid frame pairs
        for episode_dir in tqdm(episode_dirs, desc="Loading episodes"):
            frame_files = glob.glob(os.path.join(episode_dir, '*.png'))
            frame_files.sort()
            
            # Create pairs of consecutive frames
            for i in range(len(frame_files) - 1):
                frame_t = frame_files[i]
                frame_t_plus_1 = frame_files[i + 1]
                self.frame_pairs.append((frame_t, frame_t_plus_1))
        
        print(f"Total frame pairs: {len(self.frame_pairs)}")
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        frame_t_path, frame_t_plus_1_path = self.frame_pairs[idx]
        
        # Load images
        frame_t = Image.open(frame_t_path).convert('RGB')
        frame_t_plus_1 = Image.open(frame_t_plus_1_path).convert('RGB')
        
        if self.transform:
            frame_t = self.transform(frame_t)
            frame_t_plus_1 = self.transform(frame_t_plus_1)
        
        return frame_t, frame_t_plus_1

# --- Helper Functions ---
def load_vqvae_model(checkpoint_path, device):
    """Load the VQ-VAE model from checkpoint."""
    config = EvaluationConfig()
    model = VQVAE(
        input_channels_per_frame=3,
        embedding_dim=config.embedding_dim,
        num_embeddings=config.latent_vocab_size
    ).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ VQ-VAE model loaded from {checkpoint_path}")
        return model
    except FileNotFoundError:
        print(f"✗ Checkpoint not found at {checkpoint_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def create_data_loader(data_dir, batch_size, num_workers, max_episodes=None):
    """Create DataLoader for the evaluation dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [-1, 1] to match VQ-VAE training
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = FramePairDataset(data_dir, transform=transform, max_episodes=max_episodes)
    
    if len(dataset) == 0:
        print("✗ No data found in the specified directory")
        return None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def evaluate_model(model, dataloader, device, max_batches=None):
    """Evaluate the model and return average PSNR and SSIM."""
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (frame_t, frame_t_plus_1) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
                
            frame_t = frame_t.to(device)
            frame_t_plus_1 = frame_t_plus_1.to(device)
            
            # Forward pass through VQ-VAE
            output = model(frame_t, frame_t_plus_1)
            reconstructed_frame_tp1, vq_loss, _, _, perplexity, latents_e, min_encoding_indices, _ = output
            
            # Calculate metrics for each sample in the batch
            batch_size = frame_t.size(0)
            for i in range(batch_size):
                pred = reconstructed_frame_tp1[i:i+1]
                target = frame_t_plus_1[i:i+1]
                
                # Calculate PSNR and SSIM
                psnr_val = psnr(pred, target)
                ssim_val = ssim(pred, target)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                total_samples += 1
    
    # Calculate averages
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    
    return avg_psnr, avg_ssim, total_samples

def main():
    parser = argparse.ArgumentParser(description='Evaluate VQ-VAE model')
    parser.add_argument('--checkpoint', type=str, default='model-checkpoints/ALE_MsPacman-v5_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/vqvae_ms_pacman_rgb',
                        help='Directory containing episode data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum number of episodes to evaluate (None for all)')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to evaluate (None for all)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_vqvae_model(args.checkpoint, device)
    if model is None:
        return
    
    # Create data loader
    dataloader = create_data_loader(
        args.data_dir, 
        args.batch_size, 
        args.num_workers, 
        args.max_episodes
    )
    if dataloader is None:
        return
    
    # Evaluate model
    avg_psnr, avg_ssim, total_samples = evaluate_model(
        model, 
        dataloader, 
        device, 
        args.max_batches
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.checkpoint}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*50)

if __name__ == '__main__':
    main() 