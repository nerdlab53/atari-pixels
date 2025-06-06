"""
Prepares a dataset for training the Value Model.

This script combines data from the world model dataset and the rewards log
to create a dataset of (latent_state, reward) pairs. The Value Model will
be trained to predict the reward given a latent state.
"""
import json
import argparse
from tqdm import tqdm
import os

def create_value_dataset(world_model_data_path, rewards_path, output_path):
    """
    Creates a dataset of (latent_state, reward) tuples.
    """
    print(f"Loading world model data from {world_model_data_path}...")
    with open(world_model_data_path, 'r') as f:
        world_data = json.load(f)

    print(f"Loading rewards from {rewards_path}...")
    with open(rewards_path, 'r') as f:
        rewards = json.load(f)

    # Ensure the number of rewards matches the number of transitions
    if len(world_data) > len(rewards):
        print(f"Warning: Only {len(rewards)} rewards available for {len(world_data)} transitions. Truncating.")
        world_data = world_data[:len(rewards)]
        
    print("Processing data into (latent_state, reward) tuples...")
    value_model_data = []
    
    # The reward at step `i` corresponds to the transition from `t` to `t+1`,
    # so we pair `rewards[i]` with `next_latent` from `world_data[i]`.
    for i in tqdm(range(len(world_data)), desc="Creating value tuples"):
        transition = world_data[i]
        reward = rewards[i]

        value_model_data.append({
            "latent_state": transition['next_latent'],
            "reward": reward
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
        "--output_path",
        type=str,
        default="data/mspacman/value_model_data.json",
        help="Path to save the output JSON dataset for the value model."
    )
    args = parser.parse_args()
    
    create_value_dataset(args.world_model_data_path, args.rewards_path, args.output_path)

if __name__ == "__main__":
    main() 