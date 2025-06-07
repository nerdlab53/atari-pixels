import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from atari_env import AtariBreakoutEnv

def debug_color_channels():
    os.makedirs('debug', exist_ok=True)
    
    # Get frame from environment
    env = AtariBreakoutEnv(return_rgb=True)
    obs, _ = env.reset()
    env_frame = obs / 255.0  # Normalize to [0,1]
    
    # Get frame from file
    pil_img = Image.open('data/0.png').convert('RGB')
    pil_frame = np.array(pil_img, dtype=np.float32) / 255.0
    
    # Try channel-swapped versions
    rgb_to_bgr = pil_frame[:, :, ::-1]  # Swap RGB->BGR
    
    # Display all versions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(env_frame)
    axes[0].set_title('Environment Frame (Training Data Source)')
    axes[0].axis('off')
    
    axes[1].imshow(pil_frame)
    axes[1].set_title('PNG Image (Used in neural_random_game)')
    axes[1].axis('off')
    
    axes[2].imshow(rgb_to_bgr)
    axes[2].set_title('PNG Image with Channels Reversed')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug/color_channel_comparison.png')
    plt.close()
    
    print("Saved visualization to debug/color_channel_comparison.png")
    
    # Print pixel values for better comparison
    print("\nPixel Values for a Colored Region (top row, middle):")
    print(f"Environment Frame: {env_frame[20, 80]}")
    print(f"PNG Image: {pil_frame[20, 80]}")
    print(f"PNG Image (reversed): {rgb_to_bgr[20, 80]}")
    
    # Save individual images
    Image.fromarray((env_frame * 255).astype(np.uint8)).save('debug/env_frame.png')
    Image.fromarray((pil_frame * 255).astype(np.uint8)).save('debug/png_frame.png')
    Image.fromarray((rgb_to_bgr * 255).astype(np.uint8)).save('debug/reversed_frame.png')
    print("Saved individual frames to debug/ folder")

if __name__ == "__main__":
    debug_color_channels()