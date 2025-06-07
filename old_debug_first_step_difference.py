import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from latent_action_model import ActionStateToLatentMLP, load_latent_action_model

def debug_first_step_difference():
    """Compare the exact first frame from neural_random_game with validation data"""
    os.makedirs('debug', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("Loading action state model...")
    action_model = ActionStateToLatentMLP().to(device)
    ckpt = torch.load('checkpoints/latent_action/action_state_to_latent_best.pt', map_location=device)
    action_model.load_state_dict(ckpt['model_state_dict'])
    action_model.eval()
    
    print("Loading world model...")
    world_model, _ = load_latent_action_model('checkpoints/latent_action/best.pt', device)
    world_model.to(device)
    world_model.eval()
    
    # Load the exact initial frame from neural_random_game
    print("Loading initial frame...")
    init_img = Image.open('data/0.png').convert('RGB')
    init_frame_np = np.array(init_img, dtype=np.float32) / 255.0  # (210, 160, 3)
    init_frame = torch.from_numpy(init_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 210, 160)
    print(f"Initial frame shape: {init_frame.shape}, range: [{init_frame.min().item():.4f}, {init_frame.max().item():.4f}]")
    
    # Create stacked frame input (two identical frames)
    stacked_frames = torch.cat([init_frame, init_frame], dim=1)  # (1, 6, 210, 160)
    
    # Create FIRE action (index 1)
    fire_action = torch.zeros(1, 4, device=device)
    fire_action[0, 1] = 1.0  # One-hot encoding for FIRE
    
    # Get prediction from action_model
    print("Getting prediction from action model...")
    with torch.no_grad():
        logits = action_model(fire_action, stacked_frames)
        # Shape is [1, 35, 256] - take argmax over last dimension
        pred_indices = logits.argmax(dim=-1)  # Shape: [1, 35]
        print(f"Predicted indices shape: {pred_indices.shape}")
    
    # Load step 1 frame to get the ground truth
    try:
        # First try the bulk_videos path
        step1_path = 'bulk_videos/bulk_random_agent_1/1.png'
        if not os.path.exists(step1_path):
            # Fallback to other possible locations
            step1_path = 'data/1.png'
        
        print(f"Loading step 1 frame from: {step1_path}")
        step1_img = Image.open(step1_path).convert('RGB')
        step1_frame_np = np.array(step1_img, dtype=np.float32) / 255.0
        step1_frame = torch.from_numpy(step1_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)
        print(f"Step 1 frame shape: {step1_frame.shape}, range: [{step1_frame.min().item():.4f}, {step1_frame.max().item():.4f}]")
        
        # Get ground truth latent
        print("Getting ground truth latent...")
        with torch.no_grad():
            frame0_perm = init_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160) -> (1, 3, 160, 210)
            frame1_perm = step1_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160) -> (1, 3, 160, 210)
            x = torch.cat([frame0_perm, frame1_perm], dim=1)  # (1, 6, 160, 210)
            z = world_model.encoder(x)
            _, gt_indices, _, _ = world_model.vq(z)
            # Print shape for debugging
            print(f"Ground truth indices shape before flatten: {gt_indices.shape}")
            # Flatten ground truth to match predicted shape
            gt_indices_flat = gt_indices.view(1, -1)  # Shape should be [1, 35]
            print(f"Ground truth indices shape after flatten: {gt_indices_flat.shape}")
        
        # Calculate and display error
        error_count = (pred_indices != gt_indices_flat).sum().item()
        print(f"Error count on neural_random_game first frame: {error_count}/35 ({(error_count/35)*100:.1f}% error rate)")
        
        # Reshape indices for visualization (both to 5x7)
        pred_indices_reshaped = pred_indices.view(1, 5, 7)
        gt_indices_reshaped = gt_indices_flat.view(1, 5, 7)
        
        # Decode both latents to see the difference
        print("Decoding both latents...")
        with torch.no_grad():
            # Convert indices to embeddings
            embeddings = world_model.vq.embeddings
            
            # For predicted latent
            pred_indices_device = pred_indices_reshaped.to(embeddings.weight.device)
            pred_quantized = embeddings(pred_indices_device)  # (1, 5, 7, 128)
            pred_quantized = pred_quantized.permute(0, 3, 1, 2).contiguous()  # (1, 128, 5, 7)
            
            # For ground truth latent
            gt_indices_device = gt_indices_reshaped.to(embeddings.weight.device)
            gt_quantized = embeddings(gt_indices_device)  # (1, 5, 7, 128)
            gt_quantized = gt_quantized.permute(0, 3, 1, 2).contiguous()  # (1, 128, 5, 7)
            
            # Decode with the same input frame
            frame_in = init_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160) -> (1, 3, 160, 210)
            
            # Generate next frames
            pred_next_frame = world_model.decoder(pred_quantized, frame_in)  # (1, 3, 160, 210)
            pred_next_frame = pred_next_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160)
            
            gt_next_frame = world_model.decoder(gt_quantized, frame_in)  # (1, 3, 160, 210)
            gt_next_frame = gt_next_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160)
        
        # Visualize the results
        print("Creating visualization...")
        visualize_comparison(
            init_frame, step1_frame, 
            pred_indices_reshaped, gt_indices_reshaped,
            pred_next_frame, gt_next_frame, 
            error_count
        )
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

def visualize_comparison(init_frame, step1_frame, 
                         pred_indices, gt_indices, 
                         pred_next_frame, gt_next_frame, 
                         error_count):
    """Visualize the comparison between predicted and ground truth"""
    
    # Convert tensors to numpy for visualization
    init_np = init_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    step1_np = step1_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_next_np = pred_next_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    gt_next_np = gt_next_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    pred_latent_np = pred_indices.squeeze(0).cpu().numpy()
    gt_latent_np = gt_indices.squeeze(0).cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Top row
    axes[0, 0].imshow(init_np)
    axes[0, 0].set_title('Initial Frame (Input)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(step1_np)
    axes[0, 1].set_title('True Next Frame')
    axes[0, 1].axis('off')
    
    # Show latent difference mask
    diff_mask = (pred_latent_np != gt_latent_np).astype(float)
    im0 = axes[0, 2].imshow(diff_mask, cmap='Reds')
    axes[0, 2].set_title(f'Latent Differences: {error_count}/35')
    plt.colorbar(im0, ax=axes[0, 2])
    
    # Bottom row
    axes[1, 0].imshow(gt_next_np)
    axes[1, 0].set_title('Decoded GT Latent')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_next_np)
    axes[1, 1].set_title('Decoded Predicted Latent')
    axes[1, 1].axis('off')
    
    # Show latent values
    im1 = axes[1, 2].imshow(gt_latent_np, cmap='viridis')
    axes[1, 2].set_title('Ground Truth Latent vs Predicted')
    plt.colorbar(im1, ax=axes[1, 2])
    
    # Add predicted values as text overlay
    for i in range(5):
        for j in range(7):
            gt_val = gt_latent_np[i, j]
            pred_val = pred_latent_np[i, j]
            if gt_val != pred_val:
                axes[1, 2].text(j, i, f'{gt_val}→{pred_val}', 
                            ha='center', va='center', 
                            color='white', fontsize=7,
                            bbox=dict(facecolor='black', alpha=0.5))
    
    plt.suptitle('Neural Random Game Initialization: Action=FIRE\n' +
                f'Comparison of Predicted vs Ground Truth Latents ({error_count}/35 errors)', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig('debug/initial_fire_comparison.png')
    print("Visualization saved to debug/initial_fire_comparison.png")
    plt.close()
    
    # Also save individual frames for easier comparison
    plt.figure(figsize=(8, 6))
    plt.imshow(init_np)
    plt.title('Initial Frame')
    plt.axis('off')
    plt.savefig('debug/initial_frame.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(step1_np)
    plt.title('True Next Frame (FIRE action)')
    plt.axis('off')
    plt.savefig('debug/true_next_frame.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(pred_next_np)
    plt.title('Predicted Next Frame (FIRE action)')
    plt.axis('off')
    plt.savefig('debug/predicted_next_frame.png')
    plt.close()
    
    # Save latents as separate images
    plt.figure(figsize=(10, 8))
    plt.imshow(gt_latent_np, cmap='viridis')
    plt.colorbar()
    plt.title('Ground Truth Latent (FIRE action)')
    plt.savefig('debug/gt_latent.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pred_latent_np, cmap='viridis')
    plt.colorbar()
    plt.title('Predicted Latent (FIRE action)')
    plt.savefig('debug/pred_latent.png')
    plt.close()

if __name__ == "__main__":
    debug_first_step_difference()