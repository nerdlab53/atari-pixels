"""Training script for the VQ-VAE model."""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from vqvae_model import VQVAE
from latent_action_data import AtariFramePairDataset # Using the existing dataset
from utils import setup_device_logging # Assuming a utils.py for device and logging setup

def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE model on Atari frame pairs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the VQ-VAE training data (PNG frame pairs).")
    parser.add_argument("--env_name", type=str, default="BreakoutNoFrameskip-v4", help="Name of the Atari environment data was generated from.")
    
    # Model hyperparameters
    parser.add_argument("--input_channels_per_frame", type=int, default=3, help="Number of channels per input frame (3 for RGB, 1 for grayscale).")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimensionality of the latent embeddings in VQ layer.")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Number of codebook vectors in VQ layer (K).")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment cost for VQ-VAE loss (beta).")
    parser.add_argument("--codebook_entropy_reg_weight", type=float, default=0.0, help="Weight for codebook entropy regularization (gamma). Set to 0 to disable.")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu'). Autodetected if None.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model checkpoint every N epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="vqvae_checkpoints", help="Directory to save model checkpoints.")
    
    # W&B
    parser.add_argument("--wandb_project", type=str, default="atari-vqvae", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name. Defaults to env_name-vqvae-timestamp.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")

    return parser.parse_args()

def main():
    args = parse_args()

    # Setup device, logging, and W&B
    device = setup_device_logging(args.device, args.env_name, "VQVAE_Training") # Assuming setup_device_logging handles print/logging
    if not args.disable_wandb:
        run_name = args.wandb_run_name if args.wandb_run_name else f"{args.env_name}-vqvae-{wandb.util.generate_id()}"
        wandb.init(project=args.wandb_project, name=run_name, config=args)
    
    print(f"VQ-VAE Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"  Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    print("Loading dataset...")
    # Assuming AtariFramePairDataset expects root_dir and is structured as per generate_vqvae_data.py output
    # It should handle transformations like ToTensor and normalization internally.
    # If not, we need to add a transform argument.
    # For now, let's assume it processes RGB uint8 images to float tensors in [0,1]
    train_dataset = AtariFramePairDataset(root_dir=args.data_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device != "cpu" else False
    )
    print(f"Dataset loaded. Number of training samples: {len(train_dataset)}")

    # Model
    print("Initializing VQ-VAE model...")
    model = VQVAE(
        input_channels_per_frame=args.input_channels_per_frame,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        frame_t_channels_for_film=args.input_channels_per_frame, # FiLM conditioning uses frame_t
        output_channels_decoder=args.input_channels_per_frame   # Decoder outputs frame_tp1
    ).to(device)
    
    if not args.disable_wandb:
        wandb.watch(model, log_freq=100)
    print("Model initialized.")
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # TODO: Training loop
    # TODO: Evaluation/Visualization (e.g., reconstructing a few validation samples)
    # TODO: Checkpoint saving

    print("\nStarting training...")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        epoch_perplexity = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)
        for batch_idx, (frame_t, frame_tp1) in enumerate(progress_bar):
            frame_t = frame_t.to(device)
            frame_tp1 = frame_tp1.to(device)

            optimizer.zero_grad()
            
            reconstructed_frame_tp1, vq_loss, perplexity, _, min_encoding_indices = model(frame_t, frame_tp1)
            
            total_loss, recon_loss = model.calculate_loss(
                frame_tp1_original=frame_tp1,
                reconstructed_frame_tp1=reconstructed_frame_tp1,
                vq_loss=vq_loss,
                codebook_entropy_reg_weight=args.codebook_entropy_reg_weight,
                min_encoding_indices=min_encoding_indices,
                num_embeddings=args.num_embeddings
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item() # vq_loss is a component of total_loss, but good to track separately
            epoch_perplexity += perplexity.item()

            if not args.disable_wandb:
                wandb.log({
                    "train/batch_total_loss": total_loss.item(),
                    "train/batch_recon_loss": recon_loss.item(),
                    "train/batch_vq_loss": vq_loss.item(),
                    "train/batch_perplexity": perplexity.item(),
                    "epoch": epoch,
                    "batch_idx": batch_idx
                })
            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Recon Loss": f"{recon_loss.item():.4f}",
                "VQ Loss": f"{vq_loss.item():.4f}",
                "Perplexity": f"{perplexity.item():.2f}"
            })

        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_vq_loss = epoch_vq_loss / len(train_loader)
        avg_perplexity = epoch_perplexity / len(train_loader)

        print(f"Epoch {epoch}: Avg Total Loss: {avg_total_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg VQ Loss: {avg_vq_loss:.4f}, Avg Perplexity: {avg_perplexity:.2f}")

        if not args.disable_wandb:
            wandb.log({
                "train/epoch_total_loss": avg_total_loss,
                "train/epoch_recon_loss": avg_recon_loss,
                "train/epoch_vq_loss": avg_vq_loss,
                "train/epoch_perplexity": avg_perplexity,
                "epoch": epoch
            })
            
            # Log some reconstructed images
            if batch_idx == 0 and epoch % args.save_interval == 0: # Log from first batch of epoch
                if frame_t.shape[0] >= 4 and reconstructed_frame_tp1.shape[0] >=4: # Ensure enough images
                    wandb.log({
                        "train/epoch_reconstructions": [
                            wandb.Image(frame_t[i].cpu(), caption=f"frame_t_{i}") for i in range(4)
                        ] + [
                            wandb.Image(frame_tp1[i].cpu(), caption=f"frame_tp1_original_{i}") for i in range(4)
                        ] + [
                            wandb.Image(reconstructed_frame_tp1[i].cpu(), caption=f"frame_tp1_reconstructed_{i}") for i in range(4)
                        ],
                        "epoch": epoch
                    })


        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.env_name}_vqvae_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'loss': avg_total_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            if not args.disable_wandb:
                wandb.save(checkpoint_path) # Save to W&B artifacts

    if not args.disable_wandb:
        wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    main() 