import torch
from dqn_agent import DQNAgent

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Example input shape and number of actions (adjust as needed)
    state_shape = (8, 84, 84)  # 8 stacked frames, 84x84 resolution
    n_actions = 4  # Example: 4 possible actions

    agent = DQNAgent(n_actions=n_actions, state_shape=state_shape)
    print("Policy Network Architecture:\n", agent.policy_net)
    print("Target Network Architecture:\n", agent.target_net)
    print(f"\nPolicy Network Parameters: {count_parameters(agent.policy_net):,}")
    print(f"Target Network Parameters: {count_parameters(agent.target_net):,}")

if __name__ == "__main__":
    main() 