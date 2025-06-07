"""
Training script for the World Model.

This script trains a recurrent model (GRU) to predict the next VQ-VAE latent state
given the current latent state and the action taken.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
from tqdm import tqdm
import wandb
import json

from world_model import WorldModel

class WorldModelDataset(Dataset):
    """Dataset for loading (current_latent, action, next_latent) tuples from a JSON file."""
    def __init__(self, json_path):
        print(f"Loading dataset from {json_path}...")
        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {json_path}")
            print("Please run `create_world_model_dataset.py` first.")
            self.data = []
        
        if not self.data:
            print("Warning: Dataset is empty.")
            
        print(f"Dataset loaded with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'current_latent': torch.tensor(item['current_latent'], dtype=torch.long),
            'action': torch.tensor(item['action'], dtype=torch.long),
            'next_latent': torch.tensor(item['next_latent'], dtype=torch.long),
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Train a recurrent World Model.")
    parser.add_argument("--data_path", type=str, default="data/world_model_data.json", help="Path to the .json dataset.")
    
    # Model Hyperparameters
    parser.add_argument("--n_actions", type=int, default=9, help="Number of possible actions in the environment.")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of the latent and action embeddings.")
    parser.add_argument("--latent_seq_len", type=int, default=35, help="Sequence length of the VQ-VAE latent codes (e.g., 5*7).")
    parser.add_argument("--latent_vocab_size", type=int, default=256, help="Vocabulary size of the VQ-VAE codebook.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Size of the GRU hidden state.")
    parser.add_argument("--n_gru_layers", type=int, default=2, help="Number of layers in the GRU.")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu').")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--checkpoint_dir", type=str, default="world_model_checkpoints", help="Directory to save model checkpoints.")
    
    # W&B
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="atari-world-model", help="W&B project name.")

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
    full_dataset = WorldModelDataset(args.data_path)
    if len(full_dataset) == 0:
        print("Cannot proceed with an empty dataset. Exiting.")
        return
        
    # Adjust split for very small datasets
    if len(full_dataset) < 10:
        train_size = len(full_dataset)
        val_size = 0
    else:
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if val_size > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        val_loader = None


    # --- Model ---
    model = WorldModel(
        n_actions=args.n_actions,
        latent_dim=args.latent_dim,
        latent_seq_len=args.latent_seq_len,
        latent_vocab_size=args.latent_vocab_size,
        hidden_size=args.hidden_size,
        n_gru_layers=args.n_gru_layers
    ).to(device)
    
    if device.type != 'mps':
        model = torch.compile(model)
        
    print(f"WorldModel created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")
    
    # --- Training Setup ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # --- Training Loop ---
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0
        total_correct_preds = 0
        total_preds = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch in pbar:
            current_latent = batch['current_latent'].to(device)
            action = batch['action'].to(device)
            next_latent_target = batch['next_latent'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(current_latent, action)
            
            loss = criterion(logits.view(-1, args.latent_vocab_size), next_latent_target.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            total_correct_preds += (preds == next_latent_target).sum().item()
            total_preds += next_latent_target.numel()
            
            pbar.set_postfix({"loss": loss.item(), "accuracy": (total_correct_preds/total_preds)*100})

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = (total_correct_preds / total_preds) * 100
        
        # --- Validation Loop ---
        val_accuracy = 0
        avg_val_loss = 0
        if val_loader:
            model.eval()
            val_loss = 0
            val_correct_preds = 0
            val_total_preds = 0
            with torch.no_grad():
                for batch in val_loader:
                    current_latent = batch['current_latent'].to(device)
                    action = batch['action'].to(device)
                    next_latent_target = batch['next_latent'].to(device)
                    
                    logits = model(current_latent, action)
                    loss = criterion(logits.view(-1, args.latent_vocab_size), next_latent_target.view(-1))
                    val_loss += loss.item()
                    
                    preds = logits.argmax(dim=-1)
                    val_correct_preds += (preds == next_latent_target).sum().item()
                    val_total_preds += next_latent_target.numel()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = (val_correct_preds / val_total_preds) * 100

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if not args.disable_wandb:
            wandb.log({
                "train/loss": avg_train_loss,
                "train/accuracy": train_accuracy,
                "val/loss": avg_val_loss,
                "val/accuracy": val_accuracy,
                "epoch": epoch
            })
            
        # Save checkpoint
        if (epoch % 10 == 0) or (epoch == args.num_epochs):
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"world_model_epoch_{epoch}.pth"))

    if not args.disable_wandb:
        wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    main() 