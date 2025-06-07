"""Generate gameplay data (frames and videos) for VQ-VAE training.

This script records episodes of a trained DQN agent playing a specified Atari game.
For each episode, it saves:
1. A sequence of raw RGB frames as PNG files in an episode-specific subdirectory.
2. An MP4 video of the episode in the same subdirectory.

This data is intended for training a latent action prediction model (VQ-VAE).
"""

import os
import cv2
import numpy as np
import torch
import gymnasium as gym
import argparse
import time
from tqdm import tqdm

# Assuming dqn_agent.py and train_dqn.py (for AtariEnv) are in the same directory or accessible in PYTHONPATH
from dqn_agent import DQNAgent
from train_dqn import AtariEnv # Using the generic AtariEnv

def record_episode_and_save_frames(
    env, 
    agent, 
    episode_output_dir, 
    video_filename="episode.mp4",
    max_steps=1000, 
    fps=30,
    debug=False
):
    """
    Record a single episode, save its frames as PNGs, and save a video.

    Args:
        env: The Atari environment instance.
        agent: The trained agent instance.
        episode_output_dir: Directory to save frames and video for this episode.
        video_filename: Name for the video file.
        max_steps: Maximum steps per episode.
        fps: Frames per second for the output video.
        debug: Enable debug printing.
    """
    os.makedirs(episode_output_dir, exist_ok=True)
    
    obs, info = env.reset()
    
    # state_tensor for agent.select_action should be (1, C, H, W) and on the agent's device
    # obs from AtariEnv is (C, H, W) numpy array
    if isinstance(obs, np.ndarray):
        state_for_agent = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
    elif torch.is_tensor(obs):
        state_for_agent = obs.float().unsqueeze(0).to(agent.device) # Ensure it's float and has batch dim
    else:
        raise TypeError(f"Observation from environment must be np.ndarray or torch.Tensor, got {type(obs)}")

    # Get raw frame for saving (should be H, W, C for cv2.VideoWriter)
    # env.render() from AtariEnv with mode 'rgb_array' returns (H,W,C)
    raw_frame = env.render() 
    if raw_frame is None:
        print("Warning: env.render() returned None at the beginning of an episode. Skipping episode.")
        return 0, 0

    frame_height, frame_width = raw_frame.shape[0], raw_frame.shape[1]
    frame_size_for_video = (frame_width, frame_height) # cv2.VideoWriter expects (width, height)

    video_path = os.path.join(episode_output_dir, video_filename)
    
    # Try to find a working codec for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size_for_video)

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {video_path} with codec mp4v. Trying XVID with .avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(episode_output_dir, video_filename.replace(".mp4", ".avi"))
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size_for_video)
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer with XVID either. Frames will be saved, but no video.")
            video_writer = None

    cumulative_reward = 0
    frames_saved = 0
    
    for step_num in range(max_steps):
        # Save current raw_frame (before taking a step)
        png_path = os.path.join(episode_output_dir, f"{frames_saved:05d}.png")
        cv2.imwrite(png_path, cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)) # OpenCV expects BGR
        frames_saved += 1

        # Agent selects action
        action = agent.select_action(state_for_agent, epsilon=0.01) # Use a small epsilon for some exploration

        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        
        # Prepare next state for agent
        if isinstance(next_obs, np.ndarray):
            state_for_agent = torch.from_numpy(next_obs).float().unsqueeze(0).to(agent.device)
        elif torch.is_tensor(next_obs):
            state_for_agent = next_obs.float().unsqueeze(0).to(agent.device)
        else:
            raise TypeError(f"Observation from environment must be np.ndarray or torch.Tensor, got {type(next_obs)}")


        # Get the new raw_frame for the next iteration / video
        raw_frame = env.render()
        if raw_frame is None:
            if debug: print(f"Warning: env.render() returned None at step {step_num}. Using previous frame for video.")
            # If render fails, use the last good frame for video, but don't save a new PNG
            raw_frame = cv2.imread(os.path.join(episode_output_dir, f"{(frames_saved-1):05d}.png"))
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) # convert back to RGB if needed
            if raw_frame is None: # If reading fails too, break
                 print("Error: Failed to get current or previous frame. Stopping episode.")
                 break
        
        if video_writer and raw_frame is not None:
            video_writer.write(cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))

        if terminated or truncated:
            # Save the final frame
            if raw_frame is not None:
                png_path = os.path.join(episode_output_dir, f"{frames_saved:05d}.png")
                cv2.imwrite(png_path, cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
                frames_saved += 1
            break
            
    if video_writer:
        video_writer.release()
        if debug: print(f"Video saved to {video_path}")

    return cumulative_reward, frames_saved

def generate_data(
    model_path: str,
    env_name: str,
    output_dir: str,
    num_episodes: int,
    max_steps_per_episode: int = 1000,
    start_episode_num: int = 1,
    seed: int = 42,
    fps: int = 30,
    debug: bool = False
):
    """
    Generates gameplay data by running a trained agent on an Atari environment.
    Saves frames as PNGs and episode videos.
    """
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
    # Assuming AtariEnv is compatible and handles frame_stack correctly for observations.
    # The VQ-VAE works on single frames, so num_stack for AtariEnv observation might not be directly
    # relevant here, as we get raw_frames via env.render().
    # However, the agent was trained with stacked frames, so AtariEnv needs to provide that for the agent.
    # Let's assume a default stack size or infer if possible. For now, hardcoding 4.
    # This should ideally match the `state_shape` the agent was trained with.
    # We can try to infer this from a config or agent's state_shape if available.
    # For now, let's assume the agent's state_shape is known or a default like (4, 84, 84)
    # The AtariEnv render_mode must be 'rgb_array' to get frames for saving.
    
    # This should ideally come from the agent's config or be inferred.
    AGENT_FRAME_STACK_SIZE = 8 
    
    env = AtariEnv(game_id=env_name, num_stack=AGENT_FRAME_STACK_SIZE, seed=seed, render_mode='rgb_array')
    
    # --- Agent Initialization ---
    n_actions = env.action_space.n
    # Agent's state_shape is (num_stack, H, W)
    agent_state_shape = env.observation_space.shape 

    agent = DQNAgent(
        n_actions=n_actions,
        state_shape=agent_state_shape, 
        device=device
        # Assuming the DQNAgent doesn't need other complex params like use_lstm for this script's purpose
        # if it does, those need to be passed or configured.
    )
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}. Cannot generate data.")
        env.close()
        return

    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval() # Set to evaluation mode
    print(f"Loaded agent model from: {model_path}")

    # --- Gameplay Loop ---
    total_frames_collected = 0
    for i in range(num_episodes):
        current_episode_num = start_episode_num + i
        print(f"Recording Episode {current_episode_num}/{start_episode_num + num_episodes - 1}...")
        
        episode_dir_name = f"episode_{current_episode_num:04d}"
        episode_output_path = os.path.join(output_dir, episode_dir_name)
        
        # Reset environment with a new seed for each episode for variability
        current_seed = seed + current_episode_num 
        # env.reset(seed=current_seed) # AtariEnv's reset takes seed

        reward, frames_in_ep = record_episode_and_save_frames(
            env,
            agent,
            episode_output_path,
            max_steps=max_steps_per_episode,
            fps=fps,
            debug=debug
        )
        total_frames_collected += frames_in_ep
        print(f"Episode {current_episode_num} finished. Reward: {reward:.2f}, Frames saved: {frames_in_ep}")
        if frames_in_ep == 0:
            print(f"Warning: Episode {current_episode_num} saved 0 frames. Check environment rendering.")

    env.close()
    print(f"\nData generation complete. Total episodes: {num_episodes}, Total frames collected: {total_frames_collected}")
    print(f"Data saved in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate gameplay frames and videos for VQ-VAE training.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained DQN .pth model file.")
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5", help="Name of the Atari environment (e.g., 'ALE/MsPacman-v5').")
    parser.add_argument("--output_dir", type=str, default="data/vqvae_dataset", help="Directory to save recorded episodes (frames and videos).")
    parser.add_argument("--num_episodes", type=int, default=10, help="Total number of episodes to record.")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Maximum number of steps per episode.")
    parser.add_argument("--start_episode_num", type=int, default=1, help="Starting number for episode directories (e.g., episode_0001).")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for the environment. Episode seeds will be offset from this.")
    parser.add_argument("--fps", type=int, default=30, help="FPS for the output videos.")
    parser.add_argument("--debug", action='store_true', help="Enable debug printing.")
    
    args = parser.parse_args()

    generate_data(
        model_path=args.model_path,
        env_name=args.env_name,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        start_episode_num=args.start_episode_num,
        seed=args.seed,
        fps=args.fps,
        debug=args.debug
    ) 