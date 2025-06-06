"""
Training script for the Action-to-Latent Mapping Model.

This script trains a simple MLP to map a game action to the most likely
VQ-VAE latent code sequence that results from that action.
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

from latent_action_model import ActionToLatentMLP

class ActionLatentPairDataset(Dataset):
    """Dataset for loading the (action, latent_code) pairs from a JSON file."""
    def __init__(self, json_path, n_actions=9):
        print(f"Loading dataset from {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.n_actions = n_actions
        print("Dataset loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        action = item['action']
        latent_code = item['latent_code']
        
        # One-hot encode the action
        action_one_hot = torch.zeros(self.n_actions, dtype=torch.float32)
        action_one_hot[action] = 1.0
        
        return {
            'action_one_hot': action_one_hot,
            'latent_code': torch.tensor(latent_code, dtype=torch.long)
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Action-to-Latent mapping model.")
    parser.add_argument("--data_path", type=str, default="data/actions/action_latent_pairs.json", help="Path to the .json dataset.")
    parser.add_argument("--n_actions", type=int, default=9, help="Number of possible actions in the environment (for one-hot encoding).")
    
    # Model Hyperparameters from VQ-VAE
    parser.add_argument("--latent_vocab_size", type=int, default=256, help="Vocabulary size of the VQ-VAE codebook.")
    parser.add_argument("--latent_seq_len", type=int, default=35, help="Sequence length of the VQ-VAE latent codes (e.g., 5*7).")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu').")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--checkpoint_dir", type=str, default="action_mapping_checkpoints", help="Directory to save model checkpoints.")
    
    # W&B
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="atari-action-to-latent", help="W&B project name.")

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
    full_dataset = ActionLatentPairDataset(args.data_path, n_actions=args.n_actions)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model ---
    model = ActionToLatentMLP(
        n_actions=args.n_actions,
        latent_seq_len=args.latent_seq_len,
        latent_vocab_size=args.latent_vocab_size
    ).to(device)
    # model = torch.compile(model) # Disabled due to a bug in the Metal backend for torch.compile
    print(f"ActionToLatentMLP created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")
    
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
            action = batch['action_one_hot'].to(device)
            latent_target = batch['latent_code'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(action)
            loss = criterion(logits.view(-1, args.latent_vocab_size), latent_target.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            total_correct_preds += (preds == latent_target).sum().item()
            total_preds += latent_target.numel()
            
            pbar.set_postfix({"loss": loss.item(), "accuracy": (total_correct_preds/total_preds)*100})

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = (total_correct_preds / total_preds) * 100
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        val_correct_preds = 0
        val_total_preds = 0
        with torch.no_grad():
            for batch in val_loader:
                action = batch['action_one_hot'].to(device)
                latent_target = batch['latent_code'].to(device)
                
                logits = model(action)
                loss = criterion(logits.view(-1, args.latent_vocab_size), latent_target.view(-1))
                val_loss += loss.item()
                
                preds = logits.argmax(dim=-1)
                val_correct_preds += (preds == latent_target).sum().item()
                val_total_preds += latent_target.numel()
        
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
        if (epoch % 5 == 0) or (epoch == args.num_epochs):
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"action_mapping_model_epoch_{epoch}.pth"))

    if not args.disable_wandb:
        wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    main() 