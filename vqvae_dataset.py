"""PyTorch Dataset and DataLoader for VQ-VAE training on Atari frame pairs."""

import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Callable

class AtariFramePairDataset(Dataset):
    """Dataset for loading (frame_t, frame_t+1) pairs from Atari gameplay recordings."""
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        grayscale: bool = True,
        transform: Optional[Callable] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
    ):
        """
        Args:
            root_dir: Directory containing episode folders (e.g., 'data/vqvae_ms_pacman_rgb/').
            split: 'train', 'val', or 'test'.
            grayscale: If True, convert frames to grayscale. Otherwise, keep as RGB.
            transform: Optional transform to be applied on a sample.
            split_ratios: Tuple for (train_ratio, val_ratio, test_ratio).
            seed: Random seed for shuffling episodes for reproducible splits.
        """
        self.root_dir = root_dir
        self.grayscale = grayscale
        self.transform = transform
        self.split = split

        if not (sum(split_ratios) > 0.99 and sum(split_ratios) < 1.01):
            raise ValueError(f"Split ratios must sum to 1, got {split_ratios} (sum={sum(split_ratios)})")

        # Get all episode directories
        all_episode_dirs = sorted([
            os.path.join(self.root_dir, d)
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d.startswith("episode_")
        ])
        if not all_episode_dirs:
            raise FileNotFoundError(f"No episode directories found in {self.root_dir}. Ensure data is generated.")

        # Shuffle episodes for splitting
        random.Random(seed).shuffle(all_episode_dirs)

        # Split episode directories
        n_episodes = len(all_episode_dirs)
        n_train = int(n_episodes * split_ratios[0])
        n_val = int(n_episodes * split_ratios[1])
        # n_test = n_episodes - n_train - n_val # The rest go to test

        if split == 'train':
            self.episode_dirs = all_episode_dirs[:n_train]
        elif split == 'val':
            self.episode_dirs = all_episode_dirs[n_train : n_train + n_val]
        elif split == 'test':
            self.episode_dirs = all_episode_dirs[n_train + n_val :]
        else:
            raise ValueError(f"Invalid split name: {split}. Choose from 'train', 'val', 'test'.")
        
        if not self.episode_dirs:
            print(f"Warning: No episodes found for split '{self.split}' in {self.root_dir}. This might be due to too few total episodes for the split ratios.")

        # Collect all frame pairs (frame_t_path, frame_t+1_path)
        self.frame_pairs = self._collect_frame_pairs()
        if not self.frame_pairs:
             print(f"Warning: No frame pairs found for split '{self.split}'. Check episode content and paths.")


    def _collect_frame_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for ep_dir in self.episode_dirs:
            # Expecting PNG files like 00000.png, 00001.png, ...
            frame_files = sorted(
                glob.glob(os.path.join(ep_dir, '*.png')),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) # Sort by frame number
            )
            # Create (frame_t, frame_t+1) pairs
            for i in range(len(frame_files) - 1):
                pairs.append((frame_files[i], frame_files[i+1]))
        return pairs

    def _load_frame(self, path: str) -> torch.Tensor:
        """Loads a single frame from path, converts to tensor, and normalizes."""
        try:
            img = Image.open(path)
            if self.grayscale:
                img = img.convert('L')  # Convert to grayscale (1 channel)
            else:
                img = img.convert('RGB') # Ensure RGB (3 channels)
        except FileNotFoundError:
            print(f"Error: Frame not found at {path}")
            # Return a dummy tensor or handle appropriately
            return torch.zeros((1 if self.grayscale else 3, 160, 210), dtype=torch.float32) 
        except Exception as e:
            print(f"Error loading frame {path}: {e}")
            return torch.zeros((1 if self.grayscale else 3, 160, 210), dtype=torch.float32)

        # Convert to numpy array and then to tensor
        img_np = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_np /= 255.0
        
        # Reshape if grayscale: (H, W) -> (1, H, W)
        if self.grayscale and img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=0)
        # Reshape for RGB: (H, W, C) -> (C, H, W)
        elif not self.grayscale and img_np.ndim == 3:
            img_np = np.transpose(img_np, (2, 0, 1))
        
        img_tensor = torch.from_numpy(img_np)
        return img_tensor

    def __len__(self) -> int:
        return len(self.frame_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_t_path, frame_tp1_path = self.frame_pairs[idx]

        frame_t = self._load_frame(frame_t_path)
        frame_tp1 = self._load_frame(frame_tp1_path)

        # Item to return: (frame_t, frame_t+1)
        sample = {
            'frame_t': frame_t,
            'frame_tp1': frame_tp1
        }

        if self.transform:
            sample = self.transform(sample)
            
        return sample['frame_t'], sample['frame_tp1']

if __name__ == '__main__':
    # Example Usage:
    data_dir = 'data/vqvae_ms_pacman_rgb' # Adjust to your actual data path
    
    print(f"Checking dataset instantiation for directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist. Please generate data first using generate_vqvae_data.py")
        exit()

    print("\n--- Testing Train Split (RGB) ---")
    try:
        train_dataset_rgb = AtariFramePairDataset(root_dir=data_dir, split='train', grayscale=False)
        if len(train_dataset_rgb) > 0:
            print(f"Train dataset (RGB) created. Number of pairs: {len(train_dataset_rgb)}")
            frame_t, frame_tp1 = train_dataset_rgb[0]
            print(f"Sample frame_t shape: {frame_t.shape}, dtype: {frame_t.dtype}") # Expected: (3, H, W)
            print(f"Sample frame_tp1 shape: {frame_tp1.shape}, dtype: {frame_tp1.dtype}")
            # Check normalization
            print(f"Sample frame_t min: {frame_t.min()}, max: {frame_t.max()}")
        else:
            print("Train dataset (RGB) is empty. Check data and split ratios.")
    except Exception as e:
        print(f"Error creating/accessing RGB train dataset: {e}")

    print("\n--- Testing Validation Split (Grayscale) ---")
    try:
        val_dataset_gray = AtariFramePairDataset(root_dir=data_dir, split='val', grayscale=True)
        if len(val_dataset_gray) > 0:
            print(f"Validation dataset (grayscale) created. Number of pairs: {len(val_dataset_gray)}")
            g_frame_t, g_frame_tp1 = val_dataset_gray[0]
            print(f"Sample grayscale frame_t shape: {g_frame_t.shape}, dtype: {g_frame_t.dtype}") # Expected: (1, H, W)
            print(f"Sample grayscale frame_tp1 min: {g_frame_t.min()}, max: {g_frame_t.max()}")
        else:
            print("Validation dataset (grayscale) is empty.")
    except Exception as e:
        print(f"Error creating/accessing grayscale validation dataset: {e}")

    print("\n--- Testing DataLoader (RGB Train) ---")
    try:
        if len(train_dataset_rgb) > 0:
            train_dataloader = DataLoader(train_dataset_rgb, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 for easier debugging
            batch_frame_t, batch_frame_tp1 = next(iter(train_dataloader))
            print(f"Batch frame_t shape: {batch_frame_t.shape}") # Expected: (4, 3, H, W)
            print(f"Batch frame_tp1 shape: {batch_frame_tp1.shape}")
        else:
            print("Skipping DataLoader test as RGB train dataset is empty.")
    except Exception as e:
        print(f"Error with DataLoader: {e}")

    print("\nDataset tests finished. Check output carefully.") 