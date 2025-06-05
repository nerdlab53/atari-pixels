"""Record gameplay videos of a trained DQN agent playing Atari games.

This script records videos of a trained DQN agent playing a specified Atari game.
It uses Gymnasium's RecordVideo wrapper for robust video creation.

Usage:
    # Record Ms. Pac-Man using a specific model
    python record_videos.py --model_path checkpoints/MsPacman_ep100.pth --env_name "ALE/MsPacman-v5"

    # Record Breakout using a specific model
    python record_videos.py --model_path checkpoints/Breakout_ep500.pth --env_name "ALE/Breakout-v5" --output_dir videos_breakout

    # Record more episodes
    python record_videos.py --model_path checkpoints/MsPacman_ep100.pth --num_episodes 10

    # Specify a different seed
    python record_videos.py --model_path checkpoints/MsPacman_ep100.pth --seed 123

Arguments:
    --model_path (required): Path to the trained .pth model file.
    --env_name: Name of the Atari environment (e.g., 'ALE/MsPacman-v5', 'ALE/Breakout-v5'). 
                Default: 'ALE/MsPacman-v5'. Ensure this matches the game the model was trained on.
    --output_dir: Directory to save videos. If not specified, it defaults to 'videos_<game_name>'.
    --num_episodes: Number of episodes to record (default: 3).
    --seed: Base seed for the environment (default: 42).

Output:
    - Videos are saved in the specified output directory, typically named based on the 
      environment and model.
"""

import os
# import cv2 # cv2 might not be needed if RecordVideo handles all rendering/saving
import numpy as np
import torch
# from atari_env import AtariBreakoutEnv # Removed, using generic AtariEnv
from dqn_agent import DQNAgent
# from random_agent import RandomAgent # RandomAgent was part of old functions
import gymnasium as gym
import argparse
# import time # time might not be needed if old functions are removed
# import shutil # shutil might not be needed
# import tempfile # tempfile might not be needed
from gymnasium.wrappers import RecordVideo
from train_dqn import AtariEnv, config as default_config # AtariEnv is generic

# The old functions add_text_overlay, record_episode, record_gameplay_videos, 
# and record_bulk_videos have been removed as they were Breakout-specific and
# we are now relying on the more generic `record_gameplay` function below,
# which uses Gymnasium's RecordVideo wrapper.

def record_gameplay(model_path, env_name, output_dir, num_episodes=5, seed=42):
    """
    Records gameplay of a trained non-LSTM DQN agent.

    Args:
        model_path (str): Path to the trained .pth model file.
        env_name (str): Name of the Atari environment (e.g., 'ALE/MsPacman-v5').
        output_dir (str): Directory to save the recorded videos.
        num_episodes (int): Number of episodes to record.
        seed (int): Environment seed.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Environment Setup ---
    # Infer frame_stack_size from default_config if possible, otherwise use a sensible default.
    # This assumes the model was trained with settings compatible with default_config's state_shape.
    state_shape_config = default_config.get('state_shape', (4, 84, 84)) # Default to 4 if not in config
    frame_stack_size = state_shape_config[0]
    
    # Ensure we use the render_mode that is compatible with RecordVideo
    # Typically, 'rgb_array' is needed for RecordVideo to capture frames.
    # The AtariEnv from train_dqn should handle this correctly if render_mode is passed.
    eval_env = AtariEnv(game_id=env_name, num_stack=frame_stack_size, seed=seed + 1000, render_mode='rgb_array') 

    # Wrap with RecordVideo
    # Customize name_prefix to include game name and model name for clarity
    game_name_simple = env_name.split('/')[-1].replace('-v5', '')
    model_file_name = os.path.basename(model_path).replace('.pth', '')
    video_name_prefix = f"{game_name_simple}-{model_file_name}"

    video_env = RecordVideo(
        eval_env,
        video_folder=output_dir,
        episode_trigger=lambda episode_id: True, # Record every episode
        name_prefix=video_name_prefix
    )
    print(f"Recording videos to: {output_dir} with prefix {video_name_prefix}")

    # --- Agent Initialization ---
    # n_actions should come from the wrapped environment (video_env) or eval_env
    n_actions = eval_env.action_space.n 
    # state_shape for the agent should match what the AtariEnv provides
    # AtariEnv with FrameStack returns shape (num_stack, H, W)
    agent_state_shape = eval_env.observation_space.shape 

    agent = DQNAgent(
        n_actions=n_actions,
        state_shape=agent_state_shape, # Use the actual observation space shape
        device=device,
        # Assuming the agent being loaded is a standard DQN (non-LSTM, no NoisyNets for now based on user's intent to revert)
        # If the agent structure changes (e.g. back to NoisyNets), this might need adjustment or 
        # the DQNAgent constructor might need to be more flexible.
    )
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval() # Set to evaluation mode
    print(f"Loaded model from: {model_path}")
    print(f"Agent configured for state shape {agent_state_shape} and {n_actions} actions.")


    # --- Gameplay Loop ---
    for i in range(num_episodes):
        obs, info = video_env.reset(seed=seed + 1000 + i) # Ensure different seed per episode
        # obs from FrameStack is already a numpy array of the stacked frames.
        # DQNAgent.select_action expects a PyTorch tensor.
        state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
        terminated = False
        truncated = False
        total_reward = 0
        episode_length = 0

        print(f"Starting recording for episode {i+1}/{num_episodes}...")
        while not (terminated or truncated):
            # For evaluation, usually a small epsilon or greedy action.
            # This will use epsilon-greedy if the agent is reverted to standard DQN.
            # If NoisyNets are used, policy_net.eval() handles noise disabling.
            action = agent.select_action(state_tensor, epsilon=0.01) # Small epsilon for some exploration

            next_obs, reward, terminated, truncated, info = video_env.step(action)
            
            next_state_tensor = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
            state_tensor = next_state_tensor
            
            total_reward += reward
            episode_length +=1
            
            # video_env.render() # Not strictly necessary as RecordVideo handles captures

        print(f"Episode {i+1} finished. Reward: {total_reward}, Length: {episode_length}")

    video_env.close() # Important to save the last video
    # eval_env.close() # Underlying env is closed by RecordVideo wrapper
    print("Video recording finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record gameplay of a trained DQN agent for Atari games.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5", help="Name of the Atari environment (e.g., 'ALE/MsPacman-v5'). Ensure this matches the game the model was trained on.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save recorded videos. Defaults to 'videos_<game_name>'.")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to record.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for the environment.")
    
    args = parser.parse_args()

    game_name_simple = args.env_name.split('/')[-1].replace('-v5', '')
    if args.output_dir is None:
        args.output_dir = f"videos_{game_name_simple.lower()}"

    # The previous warning about MsPacman-specific setup is removed as the script is now more generic.
    # It's important that the loaded model matches the specified --env_name.

    record_gameplay(
        model_path=args.model_path,
        env_name=args.env_name,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        seed=args.seed
    )