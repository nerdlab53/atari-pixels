"""
Runs a trained agent to play Ms. Pac-Man autonomously or allows a human player to control the game.

This script loads all the trained model components for agent mode:
1. The VQ-VAE (The Eye): To perceive the world.
2. The World Model (The Dreamer): To imagine the future.
3. The Value Model (The Judge): To evaluate those futures.

It then executes the "Imagine & Evaluate" loop to select the best action
at each step of the game.

---

### How the Agent Plays Ms. Pac-Man

The agent's ability to play Ms. Pac-Man is a result of three specialized neural networks working in concert:

1.  **The VQ-VAE (The Eye):** This model acts as the agent's visual system. Its primary job is to take raw game frames (the pixels on the screen) and compress them into a much simpler, abstract representation. It learns a "vocabulary" of common visual patterns in the game. Instead of dealing with thousands of pixels, the agent can work with a short sequence of "codebook" indices from this vocabulary. This is the agent's *perception* of the world.

2.  **The World Model (The Dreamer):** This is a predictive model that acts as the agent's imagination. It takes the compressed representation from the VQ-VAE and an action (e.g., "go up"), and predicts what the world will look like *next*. It doesn't predict the full next frame in pixels, but rather the compressed representation of the next frame. By repeatedly feeding its own predictions back into itself, it can "dream" or "imagine" entire future sequences of gameplay for different action plans.

3.  **The Value Model (The Judge):** After the World Model imagines a future, the Value Model's job is to evaluate how good that future is. It looks at an imagined sequence of states and predicts the total future reward the agent is likely to get. A future where Ms. Pac-Man eats a power pellet and ghosts is good (high value), while a future where she gets cornered is bad (low value).

**The "Imagine & Evaluate" Loop:**

At each step of the game, the agent performs the following loop:
1.  **Perceive:** The VQ-VAE looks at the current screen and creates a compressed latent state.
2.  **Imagine:** For every possible action (up, down, left, right, etc.), the World Model imagines the immediate future that would result from taking that action.
3.  **Evaluate:** The Value Model looks at each of these imagined one-step futures and predicts its value (the expected future reward).
4.  **Act:** The agent chooses the action that led to the imagined future with the highest predicted value and executes it in the real game.

This cycle repeats, allowing the agent to constantly plan and re-plan, always trying to steer the game toward the most promising futures it can imagine.

---
Player Controls (when using --mode player):
- Arrow Keys: Move Ms. Pac-Man
- W, A, S, D: Alternative movement keys
- Escape: Quit the game
"""
import torch
import gymnasium as gym
import argparse
import json
from PIL import Image
from torchvision import transforms
import time
import numpy as np

# --- Import pygame for player-dream mode rendering ---
try:
    import pygame
except ImportError:
    print("Pygame not found. Please install it with 'pip install pygame' for player mode.")
    pygame = None

# --- Import Model Architectures ---
from vqvae_model import VQVAE
from world_model import WorldModel
from value_model import ValueModel

def load_models(args, device):
    """Loads all three trained models and the reward normalization stats."""
    
    # 1. Load VQ-VAE
    print("Loading VQ-VAE model...")
    vqvae_model = VQVAE(
        input_channels_per_frame=3,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.latent_vocab_size
    ).to(device)
    vqvae_model.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    vqvae_model.eval()

    # 2. Load World Model
    print("Loading World Model...")
    world_model = WorldModel(
        n_actions=args.num_actions,
        latent_dim=128,
        latent_seq_len=args.latent_seq_len,
        latent_vocab_size=args.latent_vocab_size,
        hidden_size=512,
        n_gru_layers=2
    ).to(device)
    
    # Clean the state dict if it was saved from a compiled model
    world_model_state_dict = torch.load(args.world_model_checkpoint, map_location=device)
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

    # --- Conditional Loading for Agent Mode ---
    value_model = None
    reward_stats = None

    # Only load value model and stats if the checkpoint is provided (i.e., for agent mode)
    if args.value_model_checkpoint:
        # 3. Load Value Model
        print("Loading Value Model...")
        value_model = ValueModel(
            latent_seq_len=args.latent_seq_len,
            latent_vocab_size=args.latent_vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=512
        ).to(device)
        
        # Clean the state dict for the value model as well, just in case
        value_model_state_dict = torch.load(args.value_model_checkpoint, map_location=device)
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

        # 4. Load Reward Normalization Stats
        print("Loading reward normalization stats...")
        with open(args.reward_stats_path, 'r') as f:
            reward_stats = json.load(f)
            
    print("All required models loaded.")
    return vqvae_model, world_model, value_model, reward_stats

def select_best_action(current_latent, num_actions, world_model, value_model, reward_stats, device):
    """
    Performs the "Imagine & Evaluate" loop to find the best action.
    """
    best_action = -1
    max_value = -float('inf')

    current_latent_batch = current_latent.unsqueeze(0) # Add batch dimension

    for action_idx in range(num_actions):
        with torch.no_grad():
            # Action tensor should be 1D, like in training: [batch_size]
            action_tensor = torch.tensor([action_idx], device=device) # Shape: [1]
            
            # IMAGINE: Predict the next latent state with the World Model
            imagined_next_latent_logits = world_model(current_latent_batch, action_tensor)
            imagined_next_latent_codes = torch.argmax(imagined_next_latent_logits, dim=-1)

            # EVALUATE: Predict the value of that imagined future
            predicted_normalized_value = value_model(imagined_next_latent_codes)
            
            # UN-NORMALIZE: Convert the prediction back to the true reward scale
            pred_value = predicted_normalized_value.item() * reward_stats['std'] + reward_stats['mean']
            
            if pred_value > max_value:
                max_value = pred_value
                best_action = action_idx
                
    return best_action

def main():
    parser = argparse.ArgumentParser(description="Play Ms. Pac-Man with a trained agent or yourself.")
    parser.add_argument("--mode", type=str, default="agent", choices=["agent", "player"], help="Choose 'agent' to watch the AI play, or 'player' to play yourself.")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5", help="Gymnasium environment name.")
    parser.add_argument("--num_actions", type=int, default=9, help="Number of possible actions in the game.")
    parser.add_argument("--latent_seq_len", type=int, default=35, help="Sequence length of VQ-VAE codes.")
    parser.add_argument("--latent_vocab_size", type=int, default=256, help="Vocabulary size of VQ-VAE codebook (num_embeddings).")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of each VQ-VAE embedding.")
    
    # --- Model Checkpoint Paths ---
    parser.add_argument("--vqvae_checkpoint", type=str, required=True, help="Path to the trained VQ-VAE model checkpoint.")
    parser.add_argument("--world_model_checkpoint", type=str, required=True, help="Path to the trained World Model checkpoint.")
    parser.add_argument("--value_model_checkpoint", type=str, help="Path to the trained Value Model checkpoint (only required for 'agent' mode).")
    parser.add_argument("--reward_stats_path", type=str, default="data/mspacman/reward_normalization_stats.json", help="Path to reward normalization stats.")
    parser.add_argument("--initial_frame_path", type=str, default="data/vqvae_ms_pacman_rgb/episode_0001/00000.png", help="Path to the very first frame of the game to initialize the dream.")

    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Mode-Specific Setup ---
    if args.mode == 'agent':
        if not args.value_model_checkpoint:
            parser.error("--value_model_checkpoint is required for 'agent' mode.")
        run_agent_mode(args, device)
    else: # player mode
        if pygame is None:
            exit()
        run_player_dream_mode(args, device)

def run_agent_mode(args, device):
    """The main loop for when the AI agent is playing the game via emulator."""
    # Load all models for agent mode
    vqvae, world_model, value_model, reward_stats = load_models(args, device)
    
    # --- Image Transformations ---
    transform = transforms.Compose([transforms.ToTensor()])

    # --- Initialize Environment ---
    env = gym.make(args.env_name, render_mode='human') # 'human' to display the game
    
    # --- Game Loop ---
    frame_t, info = env.reset()
    frame_tm1 = frame_t # Keep track of previous frame for encoding
    
    total_score = 0
    while True:
        # --- Preprocess and Encode Current State ---
        img_t = Image.fromarray(frame_t).convert("RGB")
        img_tm1 = Image.fromarray(frame_tm1).convert("RGB")
        tensor_t = transform(img_t).unsqueeze(0).to(device)
        tensor_tm1 = transform(img_tm1).unsqueeze(0).to(device)

        with torch.no_grad():
            frame_t_permuted = tensor_t.permute(0, 1, 3, 2)
            frame_tm1_permuted = tensor_tm1.permute(0, 1, 3, 2)
            x = torch.cat([frame_tm1_permuted, frame_t_permuted], dim=1)
            latents_e = vqvae.encoder(x)
            _, _, _, _, _, current_latent_indices = vqvae.quantizer(latents_e)

        # --- Select Best Action ---
        best_action = select_best_action(current_latent_indices.squeeze(0), args.num_actions, world_model, value_model, reward_stats, device)
        
        # --- Step the Environment ---
        frame_tp1, reward, terminated, truncated, info = env.step(best_action)
        
        # Update frames
        frame_tm1 = frame_t
        frame_t = frame_tp1

        total_score += reward
        print(f"Action: {best_action}, Reward: {reward}, Total Score: {total_score}")

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {total_score}")
            frame_t, info = env.reset()
            frame_tm1 = frame_t
            total_score = 0
            time.sleep(2) # Pause before starting next episode
            
    env.close()

def run_player_dream_mode(args, device):
    """The main loop for when the player controls the game 'dreamed' by the models."""
    print("Starting player dream mode...")
    
    # 1. Load VQ-VAE and World Model
    print("Loading VQ-VAE and World Model...")
    # We don't need the value model or reward stats for player mode
    vqvae, world_model, _, _ = load_models(args, device)
    
    # --- Pygame Setup ---
    pygame.init()
    WINDOW_WIDTH, WINDOW_HEIGHT = 420, 480 
    # Mapping keyboard keys to game actions
    ACTION_MAP_PLAYER = {
        pygame.K_UP: 1, pygame.K_w: 1,
        pygame.K_DOWN: 4, pygame.K_s: 4,
        pygame.K_LEFT: 2, pygame.K_a: 2,
        pygame.K_RIGHT: 3, pygame.K_d: 3,
        pygame.K_SPACE: 0 # NOOP
    }
    ACTION_MEANINGS = {0: "NOOP", 1: "UP", 2: "LEFT", 3: "RIGHT", 4: "DOWN"} # For display
    FPS = 10 
    
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Ms. Pac-Man Dream")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    # --- Initial State ---
    # Load the initial frame from the specified path to seed the dream.
    try:
        initial_frame_img = Image.open(args.initial_frame_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Initial frame not found at {args.initial_frame_path}")
        print("Please ensure the path is correct. Exiting.")
        return
        
    transform = transforms.Compose([transforms.ToTensor()])
    
    # We maintain the game's state using tensors on the selected device.
    current_frame_tensor = transform(initial_frame_img).unsqueeze(0).to(device)
    last_frame_tensor = current_frame_tensor.clone()
    
    running = True
    last_action = 0 # Start with NOOP action
    last_displayed_action = "START"
    
    # --- Main Game Loop ---
    while running:
        action_taken_this_frame = False
        
        # --- Event Handling (Player Input) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in ACTION_MAP_PLAYER:
                    last_action = ACTION_MAP_PLAYER[event.key]
                    last_displayed_action = ACTION_MEANINGS.get(last_action, f"ACT {last_action}")
                    action_taken_this_frame = True

        # --- Frame Generation (The "Dream") ---
        # We only generate a new frame if the player has taken an action.
        if action_taken_this_frame:
            with torch.no_grad():
                # The models were trained on frames with permuted dimensions (W, H) instead of (H, W).
                # We must permute our tensors to match the expected input shape.
                frame_t_permuted = current_frame_tensor.permute(0, 1, 3, 2)
                frame_tm1_permuted = last_frame_tensor.permute(0, 1, 3, 2)
                
                # 1. PERCEIVE: Concatenate the last two frames and encode them into a latent representation.
                x = torch.cat([frame_tm1_permuted, frame_t_permuted], dim=1)
                latents_e = vqvae.encoder(x)
                _, _, _, _, _, current_latent_indices = vqvae.quantizer(latents_e)
                
                # 2. IMAGINE: Predict the next latent state using the World Model.
                action_tensor = torch.tensor([last_action], device=device)
                # The world model expects a flattened sequence of latent codes.
                imagined_next_latent_logits = world_model(current_latent_indices.view(1, -1), action_tensor)
                imagined_next_latent_codes = torch.argmax(imagined_next_latent_logits, dim=-1)

                # 3. DECODE: Generate the next visual frame from the imagined latent state.
                quantized_next_state = vqvae.quantizer.embedding(imagined_next_latent_codes)
                
                # Reshape the flat quantized vector back into a 2D grid for the decoder.
                quantized_grid = quantized_next_state.view(1, 5, 7, -1).permute(0, 3, 1, 2)
                
                # The decoder uses the imagined state and the current frame to predict the next frame.
                imagined_frame_permuted = vqvae.decoder(quantized_grid, frame_t_permuted)
                
                # Permute the output frame back to the standard (H, W) format for display.
                imagined_frame = imagined_frame_permuted.permute(0, 1, 3, 2)

                # --- Update State History ---
                last_frame_tensor = current_frame_tensor.clone()
                current_frame_tensor = imagined_frame.clone() # This is the crucial update step.

        # --- Rendering to Screen ---
        frame_np = current_frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np.clip(0, 1) * 255).astype(np.uint8)

        # --- FIX: Channel Order Correction ---
        # The model was likely trained on BGR images (common with OpenCV).
        # Pygame expects RGB, so we need to swap the R and B channels.
        frame_np = frame_np[..., ::-1]

        # Convert to a Pygame surface. Transpose is needed because Pygame and Numpy have different coordinate systems.
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_np, (1, 0, 2)))
        
        # Scale the frame to fit our display window.
        scaled_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT - 40))
        
        # Draw all elements to the screen.
        window.fill((0, 0, 0))
        window.blit(scaled_surface, (0, 0))
        
        info_text = f"Action: {last_displayed_action}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        window.blit(text_surface, (10, WINDOW_HEIGHT - 30))
        
        pygame.display.flip()
        
        clock.tick(FPS)

    pygame.quit()
    print("Player dream mode exited.")

if __name__ == "__main__":
    main() 