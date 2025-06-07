import json
import argparse
from tqdm import tqdm
import os

def create_world_model_dataset(input_path, output_path):
    """
    Transforms a time-series of (action, latent) pairs into a dataset of
    (current_latent, action, next_latent) transitions for training a world model.
    """
    print(f"Loading time-series data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print("Processing data into (current_latent, action, next_latent) transitions...")
    world_model_data = []
    # We iterate up to the second-to-last element because each step needs a 'next' item.
    for i in tqdm(range(len(data) - 1), desc="Creating transitions"):
        current_entry = data[i]
        next_entry = data[i+1]

        # The action from the current step leads to the next latent state.
        # This creates the "cause and effect" tuple our model needs to learn from.
        world_model_data.append({
            "current_latent": current_entry['latent_code'],
            "action": current_entry['action'],
            "next_latent": next_entry['latent_code']
        })

    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(world_model_data)} transition tuples to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(world_model_data, f, indent=2)

    print("World model dataset creation complete.")

def main():
    parser = argparse.ArgumentParser(description="Create a dataset for training a latent space World Model.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/actions/action_latent_pairs.json",
        help="Path to the input JSON file with the time-series of action-latent pairs."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/world_model_data.json",
        help="Path to save the output JSON dataset for the world model."
    )
    args = parser.parse_args()

    create_world_model_dataset(args.input_path, args.output_path)

if __name__ == "__main__":
    main() 