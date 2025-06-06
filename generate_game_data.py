"""
Generates gameplay data from an Atari environment by taking random actions.

This script initializes a Gymnasium environment for an Atari game (e.g., Ms. Pac-Man),
plays for a specified number of steps with a random agent, and records the
critical information for training our models:
- The raw pixel frame for each step.
- The action taken at each step.
- The reward received after each step.

The frames are saved as individual PNG files, while the actions and rewards
are saved as JSON lists. This data is essential for the subsequent encoding
and training of the World Model and Value Model.
"""
import gymnasium as gym
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

def generate_data(args):
    # --- Setup Directories ---
    os.makedirs(args.frames_dir, exist_ok=True)
    
    # --- Initialize Environment ---
    print(f"Initializing Gymnasium environment: {args.env_name}")
    env = gym.make(args.env_name, render_mode='rgb_array')
    print(f"Action space size: {env.action_space.n}")

    # --- Data Storage ---
    actions_log = []
    rewards_log = []
    terminals_log = []

    # --- Gameplay Loop ---
    # Reset the environment to get the initial state
    frame, info = env.reset()
    
    print(f"Generating {args.num_steps} steps of gameplay data...")
    for step in tqdm(range(args.num_steps), desc="Playing Game"):
        # Select a random action from the action space
        action = env.action_space.sample()

        # Execute the action in the environment
        next_frame, reward, terminated, truncated, info = env.step(action)

        # --- Record Data ---
        # 1. Save the resulting frame
        frame_image = Image.fromarray(next_frame)
        frame_path = os.path.join(args.frames_dir, f"frame_{step:06d}.png")
        frame_image.save(frame_path)

        # 2. Log the action taken
        actions_log.append(action.item()) # Use .item() to get a standard int

        # 3. Log the reward received
        rewards_log.append(reward)

        # 4. Log the termination status
        terminals_log.append(terminated or truncated)

        # Check if the episode has ended
        if terminated or truncated:
            # print(f"\nEpisode finished at step {step}. Resetting environment.")
            frame, info = env.reset()
        else:
            frame = next_frame

    # --- Save Logs ---
    print("Saving actions and rewards logs...")
    
    actions_path = os.path.join(os.path.dirname(args.frames_dir), 'actions.json')
    with open(actions_path, 'w') as f:
        json.dump(actions_log, f)
    print(f"Actions saved to {actions_path}")

    rewards_path = os.path.join(os.path.dirname(args.frames_dir), 'rewards.json')
    with open(rewards_path, 'w') as f:
        json.dump(rewards_log, f)
    print(f"Rewards saved to {rewards_path}")

    terminals_path = os.path.join(os.path.dirname(args.frames_dir), 'terminals.json')
    with open(terminals_path, 'w') as f:
        json.dump(terminals_log, f)
    print(f"Termination signals saved to {terminals_path}")

    env.close()
    print("\nData generation complete.")

def main():
    parser = argparse.ArgumentParser(description="Generate gameplay data from a Gymnasium Atari environment.")
    
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5", help="Name of the Gymnasium environment.")
    parser.add_argument("--num_steps", type=int, default=50000, help="Total number of steps to simulate.")
    parser.add_argument("--frames_dir", type=str, default="data/raw_frames", help="Directory to save the gameplay frames.")
    
    args = parser.parse_args()
    generate_data(args)

if __name__ == "__main__":
    main() 