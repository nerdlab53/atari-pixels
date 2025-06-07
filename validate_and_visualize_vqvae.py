import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from vqvae_model import VQVAE
from world_model import WorldModel
from value_model import ValueModel

# Configuration class to hold model parameters
class ModelConfig:
    def __init__(self):
        # Updated configurations to match the trained models
        self.embedding_dim = 64  # Changed from 128 to 64
        self.latent_vocab_size = 256  # Changed from 512 to 256
        self.num_actions = 9  # Changed from 6 to 9
        self.latent_seq_len = 35
        self.vqvae_checkpoint = 'model-checkpoints/ALE_MsPacman-v5_best.pth'
        self.world_model_checkpoint = 'model-checkpoints/world_model_epoch_100.pth'
        self.value_model_checkpoint = 'model-checkpoints/value_model_epoch_50.pth'

def load_models(device):
    """Load all required models with the corrected configurations"""
    args = ModelConfig()
    models = {}
    
    # 1. Load VQ-VAE model
    print("Loading VQ-VAE model...")
    try:
        vqvae_model = VQVAE(
            input_channels_per_frame=3,
            embedding_dim=args.embedding_dim,  # Now 64
            num_embeddings=args.latent_vocab_size  # Now 256
        ).to(device)
        
        checkpoint = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
        vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        vqvae_model.eval()
        models['vqvae'] = vqvae_model
        print(f"✓ Loaded VQVAE from {args.vqvae_checkpoint}")
        print(f"  - Embedding dim: {args.embedding_dim}")
        print(f"  - Vocab size: {args.latent_vocab_size}")
    except Exception as e:
        print(f"✗ Error loading VQVAE: {e}")

    # 2. Load World Model
    print("Loading World Model...")
    try:
        world_model = WorldModel(
            n_actions=args.num_actions,  # Now 9
            latent_dim=128,  # This might also need adjustment - check your WorldModel definition
            latent_seq_len=args.latent_seq_len,
            latent_vocab_size=args.latent_vocab_size,  # Now 256
            hidden_size=512,
            n_gru_layers=2
        ).to(device)
        
        # Clean the state dict if it was saved from a compiled model
        world_model_state_dict = torch.load(args.world_model_checkpoint, map_location=device, weights_only=False)
        if list(world_model_state_dict.keys())[0].startswith('_orig_mod.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in world_model_state_dict.items():
                name = k.replace('_orig_mod.', '')
                new_state_dict[name] = v
            world_model.load_state_dict(new_state_dict)
        else:
            world_model.load_state_dict(world_model_state_dict)

        world_model.eval()
        models['world_model'] = world_model
        print(f"✓ Loaded World Model from {args.world_model_checkpoint}")
        print(f"  - Actions: {args.num_actions}")
        print(f"  - Latent vocab size: {args.latent_vocab_size}")
    except Exception as e:
        print(f"✗ Error loading World Model: {e}")

    # 3. Load Value Model
    if args.value_model_checkpoint and os.path.exists(args.value_model_checkpoint):
        print("Loading Value Model...")
        try:
            value_model = ValueModel(
                latent_seq_len=args.latent_seq_len,
                latent_vocab_size=args.latent_vocab_size,  # Now 256
                embedding_dim=args.embedding_dim,  # Now 64
                hidden_size=512
            ).to(device)
            
            # Clean the state dict for the value model as well
            value_model_state_dict = torch.load(args.value_model_checkpoint, map_location=device, weights_only=False)
            if list(value_model_state_dict.keys())[0].startswith('_orig_mod.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in value_model_state_dict.items():
                    name = k.replace('_orig_mod.', '')
                    new_state_dict[name] = v
                value_model.load_state_dict(new_state_dict)
            else:
                value_model.load_state_dict(value_model_state_dict)

            value_model.eval()
            models['value_model'] = value_model
            print(f"✓ Loaded Value Model from {args.value_model_checkpoint}")
            print(f"  - Embedding dim: {args.embedding_dim}")
            print(f"  - Latent vocab size: {args.latent_vocab_size}")
        except Exception as e:
            print(f"✗ Error loading Value Model: {e}")
    else:
        print("Value Model checkpoint not provided or not found, skipping...")
    
    return models

def create_dummy_dataloader(batch_size=32, num_batches=10):
    """Create dummy data for testing - updated for correct action space"""
    dummy_data = []
    
    for _ in range(num_batches):
        batch_size_actual = batch_size
        
        # Create dummy actions (one-hot encoded) - now 9 actions instead of 6
        actions = torch.zeros(batch_size_actual, 9)
        action_indices = torch.randint(0, 9, (batch_size_actual,))
        actions.scatter_(1, action_indices.unsqueeze(1), 1)
        
        # Create dummy frames (6 channels for concatenated frames)
        frames = torch.rand(batch_size_actual, 6, 210, 160)
        
        # Create dummy latents (35 positions with values 0-255 instead of 0-511)
        latents = torch.randint(0, 256, (batch_size_actual, 35))
        
        dummy_data.append((actions, frames, latents))
    
    return dummy_data

def inspect_checkpoint_shapes(checkpoint_path):
    """Utility function to inspect the shapes in a checkpoint"""
    print(f"\nInspecting {checkpoint_path}:")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        for key, tensor in state_dict.items():
            if not key.startswith('_orig_mod.'):  # Skip compiled model prefixes
                print(f"  {key}: {tensor.shape}")
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")

def validate_vqvae_reconstruction():
    """
    Validate VQVAE reconstruction quality on dummy or real data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('debug', exist_ok=True)
    
    # Load models
    models = load_models(device)
    
    if 'vqvae' not in models:
        print("Error: VQVAE model not loaded, cannot proceed")
        return
    
    vqvae = models['vqvae']
    
    # Create some test data
    dummy_data = create_dummy_dataloader(batch_size=8, num_batches=3)
    
    reconstruction_errors = []
    
    # Test VQVAE reconstruction
    with torch.no_grad():
        for batch_idx, (actions, frames, latents) in enumerate(dummy_data):
            frames = frames.to(device)
            
            # Split frames into individual frames (assuming 6 channels = 2 frames of 3 channels each)
            frame_t = frames[:, :3]  # Current frame
            frame_tp1 = frames[:, 3:]  # Next frame
            
            # Test reconstruction by predicting frame_tp1 from frame_t
            try:
                # Forward pass through VQVAE: expects frame_t and frame_tp1
                output = vqvae(frame_t, frame_tp1)
                
                # VQVAE returns a tuple; the first element is the reconstructed frame
                reconstructed_frame_tp1 = output[0]
                
                # Calculate reconstruction error between original and reconstructed next frame
                recon_error = torch.mean((frame_tp1 - reconstructed_frame_tp1) ** 2).item()
                reconstruction_errors.append(recon_error)
                
                # Visualize first few examples
                if batch_idx < 2:  # Only visualize first 2 batches
                    for i in range(min(2, frame_t.size(0))):
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        
                        # Original frame (t)
                        orig_frame_t = frame_t[i].permute(1, 2, 0).cpu().numpy()
                        orig_frame_t = np.clip(orig_frame_t, 0, 1)
                        axes[0].imshow(orig_frame_t)
                        axes[0].set_title('Original Frame (t)')
                        axes[0].axis('off')
                        
                        # Original frame (t+1)
                        orig_frame_tp1 = frame_tp1[i].permute(1, 2, 0).cpu().numpy()
                        orig_frame_tp1 = np.clip(orig_frame_tp1, 0, 1)
                        axes[1].imshow(orig_frame_tp1)
                        axes[1].set_title('Original Frame (t+1)')
                        axes[1].axis('off')
                        
                        # Reconstructed frame (t+1)
                        recon_frame = reconstructed_frame_tp1[i].permute(1, 2, 0).cpu().numpy()
                        recon_frame = np.clip(recon_frame, 0, 1)
                        axes[2].imshow(recon_frame)
                        axes[2].set_title('Reconstructed Frame (t+1)')
                        axes[2].axis('off')
                        
                        plt.suptitle(f'VQVAE Reconstruction - MSE: {recon_error:.6f}')
                        plt.tight_layout()
                        plt.savefig(f'debug/vqvae_recon_batch{batch_idx}_ex{i}.png')
                        plt.close()
                        
            except Exception as e:
                print(f"Error in VQVAE forward pass: {e}")
                continue
    
    # Summarize reconstruction quality
    if reconstruction_errors:
        print(f"VQVAE Reconstruction Results:")
        print(f"Average MSE: {np.mean(reconstruction_errors):.6f}")
        print(f"Std MSE: {np.std(reconstruction_errors):.6f}")
        print(f"Min MSE: {np.min(reconstruction_errors):.6f}")
        print(f"Max MSE: {np.max(reconstruction_errors):.6f}")
        
        # Plot reconstruction error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(reconstruction_errors, bins=20, alpha=0.7)
        plt.xlabel('Reconstruction MSE')
        plt.ylabel('Count')
        plt.title('VQVAE Reconstruction Error Distribution')
        plt.savefig('debug/vqvae_reconstruction_errors.png')
        plt.close()
    else:
        print("No reconstruction errors calculated")

def analyze_vqvae_latent_space():
    """
    Analyze the VQVAE latent space and quantization
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('debug', exist_ok=True)
    
    # Load models
    models = load_models(device)
    
    if 'vqvae' not in models:
        print("Error: VQVAE model not loaded, cannot proceed")
        return
    
    vqvae = models['vqvae']
    
    # Create test data
    dummy_data = create_dummy_dataloader(batch_size=16, num_batches=2)
    
    latent_codes = []
    quantization_errors = []
    
    with torch.no_grad():
        for batch_idx, (actions, frames, latents) in enumerate(dummy_data):
            frames = frames.to(device)
            frame_t = frames[:, :3]  # First frame
            frame_tp1 = frames[:, 3:] # Second frame
            
            try:
                # Get latent representations by passing both frames to the VQVAE
                output = vqvae(frame_t, frame_tp1)

                if isinstance(output, tuple) and len(output) >= 7:
                    min_encoding_indices = output[6]
                    latent_codes.extend(min_encoding_indices.flatten().cpu().tolist())

                    # The second element is vq_loss_total
                    vq_loss = output[1]
                    quantization_errors.append(vq_loss.item())
                
                # Visualize latent space for first batch
                if batch_idx == 0:
                    try:
                        # Get encoder output by replicating the logic from VQVAE.forward
                        frame_t_permuted = frame_t[:4].permute(0, 1, 3, 2)
                        frame_tp1_permuted = frame_tp1[:4].permute(0, 1, 3, 2)
                        encoder_input = torch.cat([frame_t_permuted, frame_tp1_permuted], dim=1)
                        encoder_out = vqvae.encoder(encoder_input)
                        
                        # Visualize encoder output
                        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                        axes = axes.flatten()
                        
                        for i in range(min(4, encoder_out.size(0))):
                            # Average across channels to get 2D representation
                            latent_2d = encoder_out[i].mean(dim=0).cpu().numpy()
                            
                            im = axes[i].imshow(latent_2d, cmap='viridis')
                            axes[i].set_title(f'Encoder Output {i}')
                            plt.colorbar(im, ax=axes[i])
                        
                        plt.suptitle('VQVAE Encoder Outputs')
                        plt.tight_layout()
                        plt.savefig('debug/vqvae_encoder_outputs.png')
                        plt.close()
                    except Exception as e:
                        print(f"Error visualizing encoder outputs: {e}")
                        
            except Exception as e:
                print(f"Error in latent space analysis: {e}")
                continue
    
    # Analyze latent code distribution
    if latent_codes:
        print(f"Latent Space Analysis:")
        print(f"Total latent codes sampled: {len(latent_codes)}")
        print(f"Unique codes used: {len(set(latent_codes))}")
        print(f"Code range: {min(latent_codes)} to {max(latent_codes)}")
        
        # Plot code distribution
        plt.figure(figsize=(12, 6))
        plt.hist(latent_codes, bins=50, alpha=0.7)
        plt.xlabel('Latent Code Index')
        plt.ylabel('Frequency')
        plt.title('VQVAE Latent Code Usage Distribution')
        plt.savefig('debug/vqvae_code_distribution.png')
        plt.close()
    
    if quantization_errors:
        print(f"Average quantization loss: {np.mean(quantization_errors):.6f}")

def test_model_integration():
    """
    Test integration between different models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('debug', exist_ok=True)
    
    # Load all models
    models = load_models(device)
    
    print("Loaded models:")
    for name, model in models.items():
        print(f"  - {name}: {type(model).__name__}")
        
    # Test basic functionality of each model
    dummy_data = create_dummy_dataloader(batch_size=4, num_batches=1)
    actions, frames, latents = dummy_data[0]
    
    frames = frames.to(device)
    frame_t = frames[:, :3]
    frame_tp1 = frames[:, 3:]
    
    # Test VQVAE
    if 'vqvae' in models:
        try:
            with torch.no_grad():
                vqvae_output = models['vqvae'](frame_t, frame_tp1)
                print(f"VQVAE output shape: {vqvae_output[0].shape if isinstance(vqvae_output, tuple) else vqvae_output.shape}")
        except Exception as e:
            print(f"VQVAE test failed: {e}")
    
    # Test World Model
    if 'world_model' in models:
        try:
            with torch.no_grad():
                # This would depend on your WorldModel interface
                print("World Model loaded successfully")
        except Exception as e:
            print(f"World Model test failed: {e}")
    
    # Test Value Model  
    if 'value_model' in models:
        try:
            with torch.no_grad():
                # This would depend on your ValueModel interface
                print("Value Model loaded successfully")
        except Exception as e:
            print(f"Value Model test failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("CHECKPOINT INSPECTION")
    print("=" * 60)
    
    # First, let's inspect what's actually in the checkpoints
    config = ModelConfig()
    inspect_checkpoint_shapes(config.vqvae_checkpoint)
    inspect_checkpoint_shapes(config.world_model_checkpoint)
    inspect_checkpoint_shapes(config.value_model_checkpoint)
    
    print("\n" + "=" * 60)
    print("MODEL LOADING AND TESTING")
    print("=" * 60)
    
    print("Testing VQVAE and model loading...")
    validate_vqvae_reconstruction()
    print("\nAnalyzing VQVAE latent space...")
    analyze_vqvae_latent_space()
    print("\nTesting model integration...")
    test_model_integration()