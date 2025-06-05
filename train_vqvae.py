"""Training script for the VQ-VAE model."""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb
from tqdm import tqdm
from datetime import datetime
import random
import numpy as np
import piq

from vqvae_model import VQVAE
from latent_action_data import AtariFramePairDataset # Using the existing dataset
from utils import setup_device_logging # Assuming a utils.py for device and logging setup

def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE model on Atari frame pairs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the VQ-VAE training data (PNG frame pairs).")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5", help="Name of the Atari environment data was generated from (e.g., ALE/MsPacman-v5).")
    
    # Model hyperparameters
    parser.add_argument("--input_channels_per_frame", type=int, default=3, help="Number of channels per input frame (3 for RGB, 1 for grayscale).")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimensionality of the latent embeddings. Reduced to 64 for smaller datasets to prevent overfitting.")
    parser.add_argument("--num_embeddings", type=int, default=256, help="Number of codebook vectors in VQ layer (K).")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment cost for VQ-VAE loss (beta).")
    parser.add_argument("--dropout_p", type=float, default=0.1, help="Dropout probability for Encoder/Decoder.")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu'). Autodetected if None.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model checkpoint every N epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="vqvae_checkpoints", help="Directory to save model checkpoints.")
    
    # W&B
    parser.add_argument("--wandb_project", type=str, default="atari-vqvae-refactored", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name. Defaults to env_name-vqvae-timestamp.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Weights & Biases run ID for resuming.")
    parser.add_argument("--run_name", type=str, default="VQ-VAE", help="Name for the run.")
    parser.add_argument("--codebook_entropy_warmup_epochs", type=int, default=0, help="Epochs to warm up codebook entropy regularization.")
    parser.add_argument("--use_aux_debug_loss", action="store_true", help="Use auxiliary debug loss.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval for W&B.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping (in epochs). Set to 0 to disable.")

    # Logging & W&B
    parser.add_argument("--log_images_interval", type=int, default=2500, help="Log example reconstructions every N global steps.")
    parser.add_argument("--log_metrics_interval", type=int, default=1000, help="Log PSNR/SSIM/Histograms every N global steps.")
    parser.add_argument("--is_debug", action="store_true", help="Run in debug mode on a small subset of data.")

    return parser.parse_args()

# --- START: Expert's Solution 5: Check for unused parameters / None gradients ---
def check_gradients(model, epoch, batch_idx):
    """Checks and prints names of parameters that have None gradients."""
    none_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            none_grad_params.append(name)
    if none_grad_params:
        print(f"Epoch {epoch}, Batch {batch_idx}: Warning! The following parameters have None gradients:")
        for name in none_grad_params:
            print(f"  - {name}")
    # else:
    #     print(f"Epoch {epoch}, Batch {batch_idx}: All learnable parameters have gradients.")
# --- END: Expert's Solution 5 ---

def log_advanced_metrics(reconstructed_frame_tp1, frame_tp1, min_encoding_indices, model, global_step):
    """Calculates and logs advanced metrics like PSNR, SSIM, and codebook usage."""
    log_dict = {}
    
    # Ensure images are in [0, 1] range for metrics
    reconstructed_clamp = torch.clamp(reconstructed_frame_tp1.detach(), 0, 1)
    original_clamp = torch.clamp(frame_tp1.detach(), 0, 1)
    
    # Data range is 1.0 for normalized images
    log_dict["val/psnr"] = piq.psnr(reconstructed_clamp, original_clamp, data_range=1.0).item()
    log_dict["val/ssim"] = piq.ssim(reconstructed_clamp, original_clamp, data_range=1.0).item()
    
    # Codebook usage histogram
    if min_encoding_indices is not None:
        indices_np = min_encoding_indices.cpu().numpy().flatten()
        log_dict["val/codebook_usage"] = wandb.Histogram(
            indices_np, num_bins=model.quantizer.num_embeddings
        )

    wandb.log(log_dict, step=global_step)

def main():
    args = parse_args()

    # Setup device, logging, and W&B
    device, my_logger_instance = setup_device_logging(requested_device=args.device, run_name="VQVAE_Training_Refactored") 
    message_to_log = f"Using device: {device}"
    my_logger_instance.info(message_to_log) 
    
    wandb_run = None
    if not args.disable_wandb:
        if args.wandb_run_id:
            run_id = args.wandb_run_id
            run_name = f"{args.run_name}_resume_{run_id[:8]}"
        else:
            run_id = wandb.util.generate_id()
            run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        my_logger_instance.info(f"Initializing W&B run with name: {run_name} and ID: {run_id}")
        
        config_to_log = vars(args).copy()
        config_to_log['actual_device'] = device.type

        wandb_run = wandb.init(
            project=args.wandb_project, 
            name=run_name, 
            id=run_id, 
            config=config_to_log, 
            resume="allow" if args.wandb_run_id else None
        )
        my_logger_instance.info(f"W&B run initialized. Run URL: {wandb_run.url if wandb_run else 'N/A'}")

        # --- START: Define custom x-axes for W&B --- 
        if wandb_run:
            # Batch-level metrics use global_step as their x-axis
            wandb.define_metric("train/batch_*", step_metric="global_step")
            wandb.define_metric("train/learning_rate", step_metric="global_step") # learning_rate is also per batch step

            # Epoch-level metrics use epoch as their x-axis
            wandb.define_metric("epoch", step_metric="epoch") # Define the 'epoch' metric itself as an x-axis
            wandb.define_metric("train/epoch_*", step_metric="epoch")
            wandb.define_metric("train/epoch_reconstructions", step_metric="epoch")
        # --- END: Define custom x-axes for W&B --- 

    print(f"VQ-VAE Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"  Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    print("Loading dataset...")
    # Determine grayscale based on input_channels_per_frame argument
    use_grayscale = True if args.input_channels_per_frame == 1 else False
    if args.input_channels_per_frame not in [1, 3]:
        print(f"Warning: input_channels_per_frame is {args.input_channels_per_frame}, which is unusual. Assuming RGB (3 channels) for dataset loading unless it's 1.")
        use_grayscale = False # Default to RGB for safety if an odd number is given

    full_dataset = AtariFramePairDataset(root_dir=args.data_dir, grayscale=use_grayscale)
    
    if args.is_debug:
        logger.warning("Running in DEBUG mode. Using a small subset of the dataset.")
        # Use a small, fixed subset for reproducibility in debug mode
        dataset_subset = Subset(full_dataset, range(200))
    else:
        dataset_subset = full_dataset

    # --- START: Create Train/Validation Split ---
    validation_split = 0.15
    dataset_size = len(dataset_subset)
    indices = list(range(dataset_size))
    split_idx = int((1 - validation_split) * dataset_size)
    
    # To ensure reproducibility of splits, we can use a fixed random seed for shuffling
    random.seed(42)
    random.shuffle(indices)

    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset_subset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device != "cpu" else False
    )
    val_loader = DataLoader(
        dataset_subset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device != "cpu" else False
    )
    my_logger_instance.info(f"Dataset loaded. Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    # --- END: Create Train/Validation Split ---

    # Model
    print("Initializing VQ-VAE model...")
    model = VQVAE(
        input_channels_per_frame=args.input_channels_per_frame,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        dropout_p=args.dropout_p
    ).to(device)
    
    # Compile the model for potential speedup (requires PyTorch 2.0+)
    # May have a one-time overhead for the first few batches.
    # --- TEMPORARILY DISABLING TORCH.COMPILE FOR GRADIENT DEBUGGING ---
    # print("Compiling model with torch.compile()...")
    # try:
    #     model = torch.compile(model)
    #     print("Model compiled successfully.")
    # except Exception as e:
    #     print(f"Warning: Failed to compile model with torch.compile(): {e}. Proceeding without compilation.")
    # --- END TEMPORARILY DISABLING TORCH.COMPILE ---

    if wandb_run:
        # Call wandb.watch() AFTER model is initialized and compiled
        print("Setting up wandb.watch()...")
        try:
            wandb.watch(model, log="all", log_freq=1000)
            print("wandb.watch() setup successfully.")
        except Exception as e:
            print(f"Warning: wandb.watch() failed during setup: {e}. Proceeding without it.")

    print("Model initialized.")
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # --- START: Add LR Scheduler ---
    # Decay LR by gamma every step_size epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # --- END: Add LR Scheduler ---

    # --- START: Logic for resuming from checkpoint ---
    start_epoch = 1
    global_step = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    if args.resume_checkpoint:
        my_logger_instance.info(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        try:
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            
            # Note: DDP models save with a 'module.' prefix. Our single-GPU model doesn't.
            # Create a new state_dict without the prefix if it exists.
            model_state_dict = checkpoint['model_state_dict']
            new_state_dict = {key.replace('module.', ''): value for key, value in model_state_dict.items()}
            
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            # Important: Resume global_step if it was saved, otherwise estimate it.
            # For simplicity, we can estimate. A more robust way is to save/load it.
            # Let's assume len(train_loader) is available
            # global_step = (start_epoch - 1) * len(train_loader) # This is an approximation
            # For now, let's just warn the user if global_step isn't saved.
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
                my_logger_instance.info(f"Resumed model, optimizer, and epoch {start_epoch}. Resuming global_step from {global_step}.")
            else:
                my_logger_instance.warning("global_step not found in checkpoint. W&B batch steps will restart from 0 for this run if not a resumed W&B run.")
            
            # Also resume scheduler if it was saved
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                my_logger_instance.info("Resumed LR scheduler state.")

        except FileNotFoundError:
            my_logger_instance.error(f"Checkpoint file not found: {args.resume_checkpoint}. Starting from scratch.")
        except Exception as e:
            my_logger_instance.error(f"Error loading checkpoint: {e}. Starting from scratch.")

    first_batch_checked_gradients = False 

    print("\nStarting training...")
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        epoch_perplexity = 0
        epoch_codebook_loss = 0
        epoch_commitment_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)
        for batch_idx, (frame_t, frame_tp1) in enumerate(progress_bar):
            frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)

            optimizer.zero_grad()
            
            reconstructed_frame_tp1, vq_loss_total, codebook_loss, scaled_commitment_loss, perplexity, latents_e, min_encoding_indices, quantized_for_decoder = model(frame_t, frame_tp1)
            
            total_loss, reconstruction_loss, _, _, _, _, _ = model.calculate_loss(
                frame_tp1_original=frame_tp1, 
                reconstructed_frame_tp1=reconstructed_frame_tp1, 
                vq_loss_total_from_quantizer=vq_loss_total,
                min_encoding_indices=min_encoding_indices,
                # Hardcoding old regularization args to disable them
                codebook_entropy_reg_weight=0.0,
                quantized_for_decoder_debug=None, 
                debug_embedding_weight=None,
                use_aux_debug_loss=False
            )
            
            total_loss.backward()

            # --- START: Call check_gradients after first backward pass of first epoch ---
            if not first_batch_checked_gradients and epoch == start_epoch and batch_idx == 0:
                check_gradients(model, epoch, batch_idx)
                first_batch_checked_gradients = True
            # --- END: Call check_gradients ---

            optimizer.step()
            global_step += 1 # Increment global step
            
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += reconstruction_loss.item()
            epoch_vq_loss += vq_loss_total.item() # This is the sum of codebook and commitment losses
            epoch_codebook_loss += codebook_loss.item() # Log individual component
            epoch_commitment_loss += scaled_commitment_loss.item() # Log individual component
            epoch_perplexity += perplexity.item()

            if not args.disable_wandb: 
                log_dict = {
                    "train/batch_total_loss": total_loss.item(),
                    "train/batch_reconstruction_loss": reconstruction_loss.item(),
                    "train/batch_vq_loss_total": vq_loss_total.item(),
                    "train/batch_codebook_loss": codebook_loss.item(),
                    "train/batch_commitment_loss_scaled": scaled_commitment_loss.item(),
                    "train/batch_perplexity": perplexity.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                }
                wandb.log(log_dict, step=global_step) # Use global_step for batch logging
            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Recon Loss": f"{reconstruction_loss.item():.4f}",
                "VQ Loss": f"{vq_loss_total.item():.4f}",
                "Perplexity": f"{perplexity.item():.2f}"
            })

        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_vq_loss = epoch_vq_loss / len(train_loader)
        avg_codebook_loss = epoch_codebook_loss / len(train_loader)
        avg_commitment_loss = epoch_commitment_loss / len(train_loader)
        avg_perplexity = epoch_perplexity / len(train_loader)

        my_logger_instance.info(f"Epoch {epoch}: Avg Train Loss: {avg_total_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg VQ Loss: {avg_vq_loss:.4f}, Avg Perplexity: {avg_perplexity:.2f}")

        # --- START: Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_vq_loss = 0.0
        val_perplexity = 0.0
        with torch.no_grad():
            for frame_t, frame_tp1 in tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False):
                frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)

                reconstructed_frame_tp1, vq_loss_total, _, _, perplexity, _, min_encoding_indices, _ = model(frame_t, frame_tp1)
                
                total_loss, reconstruction_loss, _, _, _, _, _ = model.calculate_loss(
                    frame_tp1_original=frame_tp1, 
                    reconstructed_frame_tp1=reconstructed_frame_tp1, 
                    vq_loss_total_from_quantizer=vq_loss_total,
                    min_encoding_indices=min_encoding_indices,
                    # Hardcoding old regularization args to disable them
                    codebook_entropy_reg_weight=0.0,
                    quantized_for_decoder_debug=None, 
                    debug_embedding_weight=None,
                    use_aux_debug_loss=False
                )
                val_loss += total_loss.item()
                val_recon_loss += reconstruction_loss.item()
                val_vq_loss += vq_loss_total.item()
                val_perplexity += perplexity.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_vq_loss = val_vq_loss / len(val_loader)
        avg_val_perplexity = val_perplexity / len(val_loader)

        my_logger_instance.info(f"Epoch {epoch}: Avg Validation Loss: {avg_val_loss:.4f}, Avg Val Recon Loss: {avg_val_recon_loss:.4f}, Avg Val Perplexity: {avg_val_perplexity:.2f}")
        
        # Step the LR scheduler
        scheduler.step(avg_val_loss)
        my_logger_instance.info(f"Epoch {epoch}: Learning rate updated to {scheduler.get_last_lr()[0]}")
        # --- END: Validation Loop ---

        if not args.disable_wandb: 
            epoch_log_dict = {
                "train/epoch_total_loss": avg_total_loss,
                "train/epoch_reconstruction_loss": avg_recon_loss,
                "train/epoch_vq_loss_total": avg_vq_loss,
                "train/epoch_codebook_loss": avg_codebook_loss, 
                "train/epoch_commitment_loss_scaled": avg_commitment_loss, 
                "train/epoch_perplexity": avg_perplexity,
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0] # Log current LR
            }
            
            # Add validation metrics to W&B log
            epoch_log_dict.update({
                "val/epoch_total_loss": avg_val_loss,
                "val/epoch_reconstruction_loss": avg_val_recon_loss,
                "val/epoch_vq_loss_total": avg_val_vq_loss,
                "val/epoch_perplexity": avg_val_perplexity
            })

            wandb.log(epoch_log_dict) 
            
            if len(train_loader) > 0 and epoch % args.save_interval == 0: 
                if frame_t.shape[0] >= 4 and reconstructed_frame_tp1.shape[0] >=4: 
                    # Log images. The 'epoch' key must be part of this log for the x-axis.
                    wandb.log({
                        "train/epoch_reconstructions": [
                            wandb.Image(frame_t[i].cpu(), caption=f"frame_t_{i}_epoch{epoch}") for i in range(4)
                        ] + [
                            wandb.Image(frame_tp1[i].cpu(), caption=f"frame_tp1_original_{i}_epoch{epoch}") for i in range(4)
                        ] + [
                            wandb.Image(reconstructed_frame_tp1[i].cpu(), caption=f"frame_tp1_reconstructed_{i}_epoch{epoch}") for i in range(4)
                        ],
                        "epoch": epoch # Ensure epoch is logged here for x-axis alignment
                    })

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.env_name.replace('/', '_')}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
                'args': args,
                'loss': best_val_loss,
                'global_step': global_step
            }, checkpoint_path)
            my_logger_instance.info(f"New best model saved to {checkpoint_path}")
            if not args.disable_wandb:
                wandb.save(checkpoint_path) # Save to W&B artifacts

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            my_logger_instance.info(f"Early stopping triggered after {args.early_stopping_patience} epochs with no improvement.")
            break

    if not args.disable_wandb: # Re-enable final W&B call
        wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    main() 