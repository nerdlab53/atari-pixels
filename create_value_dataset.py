"""
Prepares a dataset for training the Value Model.

This script creates a dataset of (latent_state, discounted_future_reward) pairs.
It calculates the discounted sum of future rewards for each step, providing a
more stable and meaningful target for the Value Model to predict.
"""
import json
import argparse
from tqdm import tqdm
import os
import numpy as np

def create_value_dataset(world_model_data_path, rewards_path, terminals_path, output_path, discount_factor=0.99):
    """
    Creates a dataset of (latent_state, discounted_future_reward) tuples.
    """
    print(f"Loading world model data from {world_model_data_path}...")
    with open(world_model_data_path, 'r') as f:
        world_data = json.load(f)

    print(f"Loading rewards from {rewards_path}...")
    with open(rewards_path, 'r') as f:
        rewards = json.load(f)
        
    print(f"Loading terminal signals from {terminals_path}...")
    with open(terminals_path, 'r') as f:
        terminals = json.load(f)

    # Ensure data lengths match
    min_len = min(len(world_data), len(rewards), len(terminals))
    if len(world_data) != min_len or len(rewards) != min_len or len(terminals) != min_len:
        print(f"Warning: Data lengths differ. Truncating to the smallest size: {min_len}.")
        world_data = world_data[:min_len]
        rewards = rewards[:min_len]
        terminals = terminals[:min_len]

    print("Calculating discounted future rewards...")
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0.0
    for i in tqdm(reversed(range(len(rewards))), desc="Discounting Rewards"):
        if terminals[i]:
            running_add = 0.0 # Reset future rewards at the end of an episode
        running_add = rewards[i] + discount_factor * running_add
        discounted_rewards[i] = running_add
        
    print("Processing data into (latent_state, discounted_reward) tuples...")
    value_model_data = []
    for i in tqdm(range(len(world_data)), desc="Creating value tuples"):
        transition = world_data[i]
        
        value_model_data.append({
            "latent_state": transition['next_latent'],
            "reward": float(discounted_rewards[i])
        })

    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(value_model_data)} tuples to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(value_model_data, f, indent=2)
    
    print("Value model dataset creation complete.")

def main():
    parser = argparse.ArgumentParser(description="Create a dataset for the Value Model.")
    parser.add_argument(
        "--world_model_data_path",
        type=str,
        default="data/mspacman/world_model_data.json",
        help="Path to the world model dataset JSON file."
    )
    parser.add_argument(
        "--rewards_path",
        type=str,
        default="data/mspacman/rewards.json",
        help="Path to the JSON file containing the list of rewards."
    )
    parser.add_argument(
        "--terminals_path",
        type=str,
        default="data/mspacman/terminals.json",
        help="Path to the JSON file containing the list of terminal signals."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/mspacman/value_model_data.json",
        help="Path to save the output JSON dataset for the value model."
    )
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor for future rewards.")
    args = parser.parse_args()
    
    create_value_dataset(
        args.world_model_data_path, 
        args.rewards_path, 
        args.terminals_path,
        args.output_path,
        args.discount_factor
    )

if __name__ == "__main__":
    main() 