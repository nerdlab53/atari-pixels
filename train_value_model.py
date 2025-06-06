"""
Training script for the Value Model.

This script trains a simple MLP to predict the expected reward from a given
latent state, using a Mean Squared Error loss.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
from tqdm import tqdm
import wandb
import json

from value_model import ValueModel

class ValueDataset(Dataset):
    """Dataset for loading (latent_state, reward) pairs from a JSON file."""
    def __init__(self, json_path):
        print(f"Loading dataset from {json_path}...")
        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {json_path}")
            print("Please run `create_value_dataset.py` first.")
            self.data = []
        
        print(f"Dataset loaded with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'latent_state': torch.tensor(item['latent_state'], dtype=torch.long),
            'reward': torch.tensor([item['reward']], dtype=torch.float32),
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Value Model.")
    parser.add_argument("--data_path", type=str, default="data/mspacman/value_model_data.json", help="Path to the .json dataset.")
    
    # Model Hyperparameters
    parser.add_argument("--latent_seq_len", type=int, default=35, help="Sequence length of the VQ-VAE latent codes.")
    parser.add_argument("--latent_vocab_size", type=int, default=256, help="Vocabulary size of the VQ-VAE codebook.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of the latent embeddings.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Size of the MLP hidden layers.")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu').")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--checkpoint_dir", type=str, default="value_model_checkpoints", help="Directory to save model checkpoints.")
    
    # W&B
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="atari-value-model", help="W&B project name.")

    return parser.parse_args()

def main():
    args = parse_args()

    # --- Setup ---
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    print(f"Using device: {device}")
    
    if not args.disable_wandb:
        wandb.init(project=args.wandb_project, config=args)

    # --- Data ---
    full_dataset = ValueDataset(args.data_path)
    if len(full_dataset) == 0:
        print("Cannot proceed with an empty dataset. Exiting.")
        return
        
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model ---
    model = ValueModel(
        latent_seq_len=args.latent_seq_len,
        latent_vocab_size=args.latent_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size
    ).to(device)
    
    if device.type != 'mps':
        model = torch.compile(model)
        
    print(f"ValueModel created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")
    
    # --- Training Setup ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() # Use Mean Squared Error for reward prediction
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # --- Training Loop ---
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch in pbar:
            latent_state = batch['latent_state'].to(device)
            reward_target = batch['reward'].to(device)
            
            optimizer.zero_grad()
            
            predicted_reward = model(latent_state)
            loss = criterion(predicted_reward, reward_target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"mse_loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                latent_state = batch['latent_state'].to(device)
                reward_target = batch['reward'].to(device)
                
                predicted_reward = model(latent_state)
                loss = criterion(predicted_reward, reward_target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train MSE: {avg_train_loss:.6f}, Val MSE: {avg_val_loss:.6f}")

        if not args.disable_wandb:
            wandb.log({
                "train/mse_loss": avg_train_loss,
                "val/mse_loss": avg_val_loss,
                "epoch": epoch
            })
            
        # Save checkpoint
        if (epoch % 5 == 0) or (epoch == args.num_epochs):
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"value_model_epoch_{epoch}.pth"))

    if not args.disable_wandb:
        wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    main() 