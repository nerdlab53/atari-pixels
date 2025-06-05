"""Training script for the VQ-VAE model."""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from datetime import datetime

from vqvae_model import VQVAE
from latent_action_data import AtariFramePairDataset # Using the existing dataset
from utils import setup_device_logging # Assuming a utils.py for device and logging setup

def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE model on Atari frame pairs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the VQ-VAE training data (PNG frame pairs).")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5", help="Name of the Atari environment data was generated from (e.g., ALE/MsPacman-v5).")
    
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
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Weights & Biases run ID for resuming.")
    parser.add_argument("--run_name", type=str, default="VQ-VAE", help="Name for the run.")
    parser.add_argument("--codebook_entropy_warmup_epochs", type=int, default=0, help="Epochs to warm up codebook entropy regularization.")
    parser.add_argument("--use_aux_debug_loss", action="store_true", help="Use auxiliary debug loss.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval for W&B.")

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

def main():
    args = parse_args()

    # Setup device, logging, and W&B
    device, my_logger_instance = setup_device_logging(requested_device=args.device, run_name="VQVAE_Training") 
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

    train_dataset = AtariFramePairDataset(root_dir=args.data_dir, grayscale=use_grayscale)
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
            wandb.watch(model, log="parameters", log_freq=100, log_graph=False)
            print("wandb.watch() setup successfully.")
        except Exception as e:
            print(f"Warning: wandb.watch() failed during setup: {e}. Proceeding without it.")

    print("Model initialized.")
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    first_batch_checked_gradients = False 

    print("\nStarting training...")
    global_step = 0  # Initialize global step counter
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        epoch_perplexity = 0
        epoch_codebook_loss = 0
        epoch_commitment_loss = 0
        epoch_aux_debug_loss = 0
        epoch_entropy_reg_term = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)
        for batch_idx, (frame_t, frame_tp1) in enumerate(progress_bar):
            frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)

            optimizer.zero_grad()
            
            reconstructed_frame_tp1, vq_loss_total, codebook_loss, scaled_commitment_loss, perplexity, latents_e, min_encoding_indices, quantized_for_decoder = model(frame_t, frame_tp1)
            
            total_loss, reconstruction_loss, _, _, _, aux_debug_loss_val, entropy_reg_val = model.calculate_loss(
                frame_tp1_original=frame_tp1, 
                reconstructed_frame_tp1=reconstructed_frame_tp1, 
                vq_loss_total_from_quantizer=vq_loss_total,
                codebook_loss_from_quantizer=codebook_loss, # Pass through for now, though calculate_loss might not use it directly yet for total_loss
                scaled_commitment_loss_from_quantizer=scaled_commitment_loss, # Pass through
                quantized_for_decoder_debug=quantized_for_decoder, 
                codebook_entropy_reg_weight=args.codebook_entropy_reg_weight if epoch >= args.codebook_entropy_warmup_epochs else 0.0,
                min_encoding_indices=min_encoding_indices,
                debug_embedding_weight=model.quantizer.embedding.weight if args.use_aux_debug_loss else None,
                use_aux_debug_loss=args.use_aux_debug_loss
            )
            
            total_loss.backward()

            # --- START: Call check_gradients after first backward pass of first epoch ---
            if not first_batch_checked_gradients and epoch == 1 and batch_idx == 0:
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
            if aux_debug_loss_val is not None:
                epoch_aux_debug_loss += aux_debug_loss_val.item()
            if entropy_reg_val is not None:
                epoch_entropy_reg_term += entropy_reg_val.item()

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
                if args.use_aux_debug_loss and aux_debug_loss_val is not None:
                    log_dict["train/batch_aux_debug_loss"] = aux_debug_loss_val.item()
                if entropy_reg_val is not None:
                    log_dict["train/batch_entropy_reg_term"] = entropy_reg_val.item()
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
        avg_aux_debug_loss = epoch_aux_debug_loss / len(train_loader) if args.use_aux_debug_loss and epoch_aux_debug_loss > 0 else 0
        avg_entropy_reg_term = epoch_entropy_reg_term / len(train_loader) if args.codebook_entropy_reg_weight > 0.0 else 0

        my_logger_instance.info(f"Epoch {epoch}: Avg Total Loss: {avg_total_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg VQ Loss: {avg_vq_loss:.4f}, Avg Perplexity: {avg_perplexity:.2f}, Avg Entropy Term: {avg_entropy_reg_term:.4f}")

        if not args.disable_wandb: 
            epoch_log_dict = {
                "train/epoch_total_loss": avg_total_loss,
                "train/epoch_reconstruction_loss": avg_recon_loss,
                "train/epoch_vq_loss_total": avg_vq_loss,
                "train/epoch_codebook_loss": avg_codebook_loss, 
                "train/epoch_commitment_loss_scaled": avg_commitment_loss, 
                "train/epoch_perplexity": avg_perplexity,
                "epoch": epoch
            }
            if args.use_aux_debug_loss:
                epoch_log_dict["train/epoch_aux_debug_loss"] = avg_aux_debug_loss
            if args.codebook_entropy_reg_weight > 0.0:
                epoch_log_dict["train/epoch_entropy_reg_term"] = avg_entropy_reg_term
            
            # Log epoch summary. The dictionary itself contains the 'epoch' key which will be used as x-axis
            # based on define_metric("...", step_metric="epoch") for relevant keys.
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
        if epoch % args.save_interval == 0:
            # Sanitize env_name for filename by replacing slashes
            safe_env_name = args.env_name.replace("/", "_")
            checkpoint_filename = f"{safe_env_name}_vqvae_epoch_{epoch}.pth"
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)
            
            # Ensure the immediate directory for the checkpoint exists (though checkpoint_dir itself is already created)
            # For this simpler filename structure, os.makedirs(args.checkpoint_dir, exist_ok=True) is sufficient.
            # If checkpoint_filename itself contained subdirectories, we would do:
            # os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            # But with the flattened name, this is not needed beyond the initial args.checkpoint_dir creation.

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'loss': avg_total_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            if not args.disable_wandb: # Re-enable W&B artifact saving
                wandb.save(checkpoint_path) # Save to W&B artifacts

    if not args.disable_wandb: # Re-enable final W&B call
        wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    main() 