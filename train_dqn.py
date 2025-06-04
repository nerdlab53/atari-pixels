import os
import numpy as np
import torch
from tqdm import trange
import gymnasium as gym
from gymnasium.core import ObservationWrapper
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from dqn_agent import DQNAgent, ReplayBuffer
import cv2
import json
import random
import argparse
from rnd import RandomNetworkDistillation
import csv
import time
import wandb
from collections import deque

# Config
config = {
    'env_name': 'ALE/MsPacman-v5',
    'n_actions': 9,
    'state_shape': (8, 84, 84),
    'max_episodes': 2000,
    'max_steps': 1000,
    'target_update_freq': 10000, #since i'm doing 5 update steps per environment step, this means 200*5=1000 steps between target network updates
    'checkpoint_dir': 'checkpoints',
    'data_dir': 'data/raw_gameplay',
    'actions_dir': 'data/actions',
    'save_freq': 10,
    'min_buffer': 100000,
    'seed': 42,
    'epsilon': 0.1,  # Epsilon for epsilon-greedy exploration
    'per_alpha': 0.6,  # Alpha for Prioritized Experience Replay
    'per_beta_start': 0.4,  # Initial beta for PER
    'per_beta_frames': 1000000,  # Frames over which to anneal beta for PER
}

# Set seeds and deterministic flags
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
random.seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Generic Atari Environment --- #
class AtariEnv:
    def __init__(self, game_id='ALE/Breakout-v5', screen_height=84, screen_width=84, num_stack=8, seed=None, render_mode=None):
        if seed is not None:
            self.env = gym.make(game_id, obs_type="grayscale", render_mode=render_mode, full_action_space=False)
            # For full_action_space=False, Breakout has 4 actions, MsPacman has 9 actions.
            # For full_action_space=True, Breakout has 18 actions, MsPacman has 18 actions.
        else:
            self.env = gym.make(game_id, obs_type="grayscale", render_mode=render_mode, full_action_space=False)
        
        self.env = ResizeObservation(self.env, shape=(screen_height, screen_width))
        # GrayScaleObservation is applied by obs_type="grayscale" in make, FrameStack handles num_stack
        # self.env = GrayScaleObservation(self.env, keep_dim=False) # keep_dim=False removes the channel dim
        # FrameStack expects a single channel image (H, W) if keep_dim=False for GrayScaleObservation
        # or (H, W, 1) if keep_dim=True. Since obs_type="grayscale" gives (H,W), FrameStack works directly.
        
        # Note: FrameStack will stack on axis 0 by default, so (num_stack, H, W)
        self.env = FrameStack(self.env, num_stack=num_stack)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # Gymnasium's FrameStack stacks on axis 0 automatically.
        # The observation from FrameStack is already a LazyFrames object containing stacked frames.
        obs, info = self.env.reset(seed=seed, options=options)
        # No need to manually stack: obs is already (num_stack, H, W)
        return np.array(obs), info 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs is already (num_stack, H, W)
        return np.array(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode:
            return self.env.render()

    def close(self):
        self.env.close()

    def get_action_meanings(self):
        if hasattr(self.env, 'get_action_meanings'):
            return self.env.get_action_meanings()
        elif hasattr(self.env.unwrapped, 'get_action_meanings'):
            return self.env.unwrapped.get_action_meanings()
        return [f"ACTION_{i}" for i in range(self.action_space.n)]

# --- End Generic Atari Environment --- #

def save_episode_data(frames, actions, rewards, skill_level, episode_idx, config):
    """Save frames as PNGs and actions/rewards as JSON."""
    skill_dir = os.path.join(config['data_dir'], f'skill_level_{skill_level}', f'episode_{episode_idx:03d}')
    os.makedirs(skill_dir, exist_ok=True)
    actions_json = []
    for i, (frame, action, reward) in enumerate(zip(frames, actions, rewards)):
        frame_path = os.path.join(skill_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame)
        actions_json.append({'step': i, 'action': int(action), 'reward': float(reward)})
    # Save actions JSON
    actions_file = os.path.join(config['actions_dir'], f'skill_level_{skill_level}_actions.json')
    os.makedirs(config['actions_dir'], exist_ok=True)
    if os.path.exists(actions_file):
        with open(actions_file, 'r') as f:
            all_actions = json.load(f)
    else:
        all_actions = []
    all_actions.append({'episode': episode_idx, 'actions': actions_json})
    with open(actions_file, 'w') as f:
        json.dump(all_actions, f, indent=2)


def evaluate_agent(agent, env, n_episodes=10, log_id=None, return_q_values=False):
    """Evaluate agent and return average reward. Optionally return all Q-values for wandb logging."""
    rewards = []
    log_rows = []
    all_q_values = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        state_stack = np.stack([obs] * 8, axis=0)
        done = False
        total_reward = 0
        for step in range(config['max_steps']):
            state_tensor = torch.from_numpy(state_stack).unsqueeze(0).to(agent.device)
            agent.policy_net.eval()
            with torch.no_grad():
                q_values_tensor = agent.policy_net(state_tensor)
                q_values = q_values_tensor.cpu().numpy().flatten()
                if np.random.rand() < 0.05:
                    action = np.random.randint(0, config['n_actions'])
                else:
                    action = int(np.argmax(q_values))
            next_obs, reward, terminated, truncated, info = env.step(action)
            q_values_stable = q_values - np.max(q_values)
            exp_q = np.exp(q_values_stable)
            probs = exp_q / np.sum(exp_q)
            log_rows.append({
                'episode': ep,
                'step': step,
                'q_values': ','.join(f'{q:.4f}' for q in q_values),
                'probabilities': ','.join(f'{p:.4f}' for p in probs),
                'action_selected': action,
                'reward': reward
            })
            all_q_values.append(q_values)
            state_stack = np.roll(state_stack, shift=-1, axis=0)
            state_stack[-1] = next_obs
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        rewards.append(total_reward)
    # Save CSV log
    os.makedirs('eval_logs', exist_ok=True)
    if log_id is None:
        log_id = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join('eval_logs', f'eval_log_{log_id}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'step', 'q_values', 'probabilities', 'action_selected', 'reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)
    if return_q_values:
        return np.mean(rewards), np.concatenate(all_q_values)
    return np.mean(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--min_buffer', type=int, default=None)
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--no_save', action='store_true', help='Do not save frames or actions during training')
    parser.add_argument('--exploration_mode', type=str, choices=['epsilon'], default='epsilon', help='Exploration mode: epsilon-greedy (default)')
    parser.add_argument('--use_per', action='store_true', help='Use Prioritized Experience Replay and beta annealing')
    args = parser.parse_args()
    # Override config if args provided
    if args.max_episodes is not None:
        config['max_episodes'] = args.max_episodes
    if args.min_buffer is not None:
        config['min_buffer'] = args.min_buffer
    if args.save_freq is not None:
        config['save_freq'] = args.save_freq
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['actions_dir'], exist_ok=True)

    # --- Determine run suffix for naming --- #
    # Extract a short game name, e.g., MsPacman from ALE/MsPacman-v5
    short_game_name = config['env_name'].split('/')[-1].split('-')[0]
    per_suffix = "_PER" if args.use_per else "_NoPER"
    run_name_suffix = f"{short_game_name}{per_suffix}"

    # --- wandb setup ---
    wandb.init(project="atari-drl-experiments", config=config, name=f"DQN_{run_name_suffix}_{time.strftime('%Y%m%d_%H%M%S')}")

    # --- Exploration Mode Setup ---
    exploration_mode = args.exploration_mode
    if exploration_mode == 'epsilon':
        agent_params = {
            'n_actions': config['n_actions'],
            'state_shape': config['state_shape'],
            'device': device,
        }
        if args.use_per:
            agent_params['prioritized'] = True
            agent_params['per_alpha'] = config['per_alpha']
            agent_params['per_beta'] = config['per_beta_start']
            print("INFO: Using Prioritized Experience Replay with beta annealing.")
        else:
            agent_params['prioritized'] = False
            print("INFO: Using standard Experience Replay (no PER).")
        
        agent = DQNAgent(**agent_params)
        epsilon_start = 1.0
        epsilon_final = 0.1
        epsilon_decay_steps = 1_000_000
    else:
        shared_replay_buffer = ReplayBuffer(capacity=1000000)
        exploration_agent = DQNAgent(n_actions=config['n_actions'], state_shape=config['state_shape'], replay_buffer=shared_replay_buffer, device=device)
        exploitation_agent = DQNAgent(n_actions=config['n_actions'], state_shape=config['state_shape'], replay_buffer=shared_replay_buffer, device=device)
        from rnd import RandomNetworkDistillation
        rnd = RandomNetworkDistillation(state_shape=config['state_shape'], output_dim=512, lr=1e-5, reward_scale=0.1, device=device)
        alpha = 0.5
        alpha_decay = 0.99995
        min_alpha = 0.05
    # ---

    total_steps = 0
    exploration_rewards = []
    exploitation_rewards = []
    intrinsic_rewards_log = []
    running_avg_rewards = []
    pbar = trange(config['max_episodes'], desc='Training')

    epsilon = max(epsilon_final, epsilon_start - (epsilon_start - epsilon_final) * min(1.0, total_steps / epsilon_decay_steps))
    current_beta = config['per_beta_start'] if args.use_per else None # Initialize for logging

    # Fill replay buffer with random actions first
    print("Pre-filling replay buffer with random experiences...")
    env = AtariEnv(game_id=config['env_name'], num_stack=config['state_shape'][0], seed=config['seed'])
    print(f"Initialized environment: {config['env_name']}")
    print(f"Action space size: {env.action_space.n}")
    # Ensure config n_actions matches env action space
    if config['n_actions'] != env.action_space.n:
        print(f"WARNING: config n_actions ({config['n_actions']}) does not match env action space ({env.action_space.n}). Updating config.")
        config['n_actions'] = env.action_space.n
        # Re-initialize agent_params if n_actions changed for the DQNAgent
        # This is important if agent was already initialized before this check
        # However, agent is initialized after this block, so it should be fine.

    obs, info = env.reset(seed=config['seed'])
    state_stack = obs # obs from AtariEnv with FrameStack is already (num_stack, H, W)
    replay_buffer = agent.replay_buffer if exploration_mode == 'epsilon' else shared_replay_buffer
    for step in range(max(5000, config['min_buffer'])):
        action = np.random.randint(0, config['n_actions'])
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # next_state_stack = np.roll(state_stack, shift=-1, axis=0) # No longer needed
        # next_state_stack[-1] = next_obs # Error was here
        next_state_stack = next_obs # next_obs is already the new full stack

        replay_buffer.push(state_stack, action, reward, 0.0, next_state_stack, terminated or truncated)
        state_stack = next_state_stack
        if terminated or truncated:
            obs, info = env.reset(seed=config['seed'] if step < 100 else None) # Re-seed for first few resets for consistency
            state_stack = obs
        if step % 1000 == 0:
            print(f"  {step}/{max(5000, config['min_buffer'])} experiences collected")

    eval_env = AtariEnv(game_id=config['env_name'], num_stack=config['state_shape'][0], seed=config['seed']+1) # Use a different seed for eval

    window_size_for_logs = 30
    running_losses = deque(maxlen=window_size_for_logs)
    running_td_errors = deque(maxlen=window_size_for_logs)
    running_rewards = deque(maxlen=window_size_for_logs)
    log_into_wandb=False
    for episode in pbar:
        losses = []
        td_errors = []
        # Periodic evaluation
        if episode % 10 == 0:
            log_id = f"ep{episode}"
            eval_reward, q_value_dist = evaluate_agent(agent, eval_env, n_episodes=5, log_id=log_id, return_q_values=True)
            policy_str = f"Policy eval: {eval_reward:.1f}"
            log_into_wandb=True
            
        else:
            policy_str = ""

        obs, info = env.reset()
        state_stack = np.stack([obs] * 8, axis=0)
        frames, actions, extrinsic_rewards, intrinsic_rewards = [obs[-1]], [], [], [] # Store last frame of initial stack
        total_combined_reward = 0
        total_extrinsic_reward = 0
        done = False
        if exploration_mode == 'epsilon':
            if args.use_per:
                # Anneal PER beta
                beta_progress = min(1.0, total_steps / config['per_beta_frames'])
                current_beta = config['per_beta_start'] + beta_progress * (1.0 - config['per_beta_start'])
                agent.anneal_per_beta(current_beta)
            else:
                current_beta = None # Ensure it's None if PER is not used

            for step in range(config['max_steps']):
                # Epsilon annealing
                epsilon = max(epsilon_final, epsilon_start - (epsilon_start - epsilon_final) * min(1.0, total_steps / epsilon_decay_steps))
                action = agent.select_action(state_stack, mode='epsilon', epsilon=epsilon)
                next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)
                
                # next_state_stack = np.roll(state_stack, shift=-1, axis=0) # No longer needed
                # next_state_stack[-1] = next_obs # Error was here
                next_state_stack = next_obs # next_obs is already the new full stack
                
                agent.replay_buffer.push(state_stack, action, extrinsic_reward, 0.0, next_state_stack, terminated or truncated)
                for _ in range(1):
                    result = agent.optimize_model(mode='exploitation')
                    if result is not None:
                        loss, td_error = result
                        losses.append(loss)
                        td_errors.append(td_error)
                state_stack = next_state_stack
                frames.append(next_obs[-1]) # Store last frame of the new stack for visualization
                actions.append(action)
                extrinsic_rewards.append(extrinsic_reward)
                total_extrinsic_reward += extrinsic_reward
                total_combined_reward += extrinsic_reward
                total_steps += 1
                if total_steps % config['target_update_freq'] == 0:
                    agent.update_target_network()
                    print(f"Target network updated at step {total_steps}")
                if terminated or truncated:
                    break
            avg_loss = np.mean(losses) if losses else 0.0
            exploration_rewards.append(total_combined_reward)
            exploitation_rewards.append(total_extrinsic_reward)
            running_avg = np.mean(exploration_rewards[-min(window_size_for_logs, len(exploration_rewards)):])
            running_avg_rewards.append(running_avg)
            running_losses.append(avg_loss)
            running_rewards.append(total_combined_reward)
            avg_td_error = np.mean(td_errors) if td_errors else 0.0
            running_td_errors.append(avg_td_error)
            pbar.set_postfix({
                'explore_r': total_combined_reward,
                'exploit_r': total_extrinsic_reward,
                'epsilon': f"{epsilon:.3f}",
                'loss': f"{avg_loss:.3f}",
                'td_error': f"{avg_td_error:.3f}"
            })
            if episode > 0 and episode % 10 == 0:
                last_10_explore = exploration_rewards[-10:]
                last_10_exploit = exploitation_rewards[-10:]
                print(f"[Stats] Episodes {episode-9}-{episode} | Explore Avg: {np.mean(last_10_explore):.2f} | Exploit Avg: {np.mean(last_10_exploit):.2f} | {policy_str}")
                
                
                
        else:
            # --- RND mode (original logic) ---
            for step in range(config['max_steps']):
                action = exploration_agent.select_action(state_stack, mode='softmax', temperature=1.0, epsilon=config['epsilon'])
                next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)
                # next_state_stack = np.roll(state_stack, shift=-1, axis=0) # No longer needed
                # next_state_stack[-1] = next_obs # Error was here
                next_state_stack = next_obs # next_obs is already the new full stack

                intrinsic_reward = float(rnd.compute_intrinsic_reward(np.expand_dims(next_state_stack, axis=0)).cpu().numpy()[0])
                shared_replay_buffer.push(state_stack, action, extrinsic_reward, intrinsic_reward, next_state_stack, terminated or truncated)
                for _ in range(5):
                    result = exploration_agent.optimize_model(mode='exploration', alpha=alpha)
                    if result is not None:
                        loss, td_error = result
                        losses.append(loss)
                        td_errors.append(td_error)
                for _ in range(5):
                    _ = exploitation_agent.optimize_model(mode='exploitation')
                if np.random.rand() < 0.2:
                    if len(shared_replay_buffer) >= 128:
                        batch = shared_replay_buffer.sample(128, mode='exploration', alpha=alpha)
                        states, _, _, _, _ = batch
                        states_np = np.stack(states)
                        rnd.update(states_np)
                state_stack = next_state_stack
                frames.append(next_obs[-1]) # Store last frame of the new stack for visualization
                actions.append(action)
                extrinsic_rewards.append(extrinsic_reward)
                intrinsic_rewards.append(intrinsic_reward)
                total_extrinsic_reward += extrinsic_reward
                total_combined_reward += (1 - alpha) * extrinsic_reward + alpha * intrinsic_reward
                total_steps += 1
                if total_steps % config['target_update_freq'] == 0:
                    exploration_agent.update_target_network()
                    exploitation_agent.update_target_network()
                if terminated or truncated:
                    break
            avg_loss = np.mean(losses) if losses else 0.0
            exploration_rewards.append(total_combined_reward)
            exploitation_rewards.append(total_extrinsic_reward)
            intrinsic_rewards_log.append(np.sum(intrinsic_rewards))
            running_avg = np.mean(exploration_rewards[-min(window_size_for_logs, len(exploration_rewards)):])
            running_avg_rewards.append(running_avg)
            running_rewards.append(total_combined_reward)
            running_losses.append(avg_loss)
            avg_td_error = np.mean(td_errors) if td_errors else 0.0
            running_td_errors.append(avg_td_error)
            alpha = max(min_alpha, alpha * alpha_decay)
            pbar.set_postfix({
                'explore_r': total_combined_reward,
                'exploit_r': total_extrinsic_reward,
                'alpha': f"{alpha:.3f}",
                'loss': f"{avg_loss:.3f}",
                'td_error': f"{avg_td_error:.3f}"
            })
            if episode > 0 and episode % 10 == 0:
                last_10_explore = exploration_rewards[-10:]
                last_10_exploit = exploitation_rewards[-10:]
                last_10_intrinsic = intrinsic_rewards_log[-10:]
                print(f"[Stats] Episodes {episode-9}-{episode} | Explore Avg: {np.mean(last_10_explore):.2f} | Exploit Avg: {np.mean(last_10_exploit):.2f} | {policy_str} | Alpha: {alpha:.3f}")
                print(f"RND Stats: {rnd.get_stats()}")
                intrinsic_ratio = np.mean(last_10_intrinsic) / (np.mean(np.abs(last_10_exploit)) + 1e-8)
                print(f"Intrinsic/Extrinsic ratio: {intrinsic_ratio:.2f}")
        
        if log_into_wandb:
            # Log to wandb
            try:
                # Get weight and grad norms
                # Determine which agent to get norms from based on exploration mode
                current_active_agent = None
                if exploration_mode == 'epsilon':
                    current_active_agent = agent
                elif exploration_mode != 'epsilon': # RND mode
                    current_active_agent = exploitation_agent # Or exploration_agent, depending on what you want to track
                
                if current_active_agent is not None:
                    weight_norms = current_active_agent.get_weight_norms()
                    grad_norms = current_active_agent.get_grad_norms()
                else:
                    # Fallback or handle if no agent is active (should not happen in current logic)
                    weight_norms = {'policy_weight_norm': 0, 'target_weight_norm': 0, 'policy_max_weight': 0, 'policy_min_weight': 0}
                    grad_norms = {'policy_grad_norm': 0, 'target_grad_norm': 0}

                wandb.log({
                    'eval/total_steps': total_steps,
                    'eval/episode': episode,
                    'eval/reward': eval_reward,
                    'eval/q_value_mean': np.mean(q_value_dist),
                    'eval/q_value_std': np.std(q_value_dist),
                    'eval/q_value_min': np.min(q_value_dist),
                    'eval/q_value_max': np.max(q_value_dist),
                    'eval/q_value_hist': wandb.Histogram(q_value_dist),
                    'train/reward_current': running_rewards[-1] if running_rewards else 0,
                    'train/loss_current': running_losses[-1] if running_losses else 0,
                    'train/td_error_current': running_td_errors[-1] if running_td_errors else 0,
                    'train/running_reward': np.mean(running_rewards) if running_rewards else 0,
                    'train/running_loss': np.mean(running_losses) if running_losses else 0,
                    'train/running_td_error': np.mean(running_td_errors) if running_td_errors else 0,
                    'train/epsilon': epsilon if exploration_mode == 'epsilon' else None,
                    'train/per_beta': current_beta if args.use_per and exploration_mode == 'epsilon' and agent.prioritized else None,
                    'train/alpha': alpha if exploration_mode != 'epsilon' else None,
                    'train/episode': episode,
                    'policy/weight_norm': weight_norms['policy_weight_norm'],
                    'policy/grad_norm': grad_norms['policy_grad_norm'],
                    'target/weight_norm': weight_norms['target_weight_norm'],
                    'target/grad_norm': grad_norms['target_grad_norm'],
                    'policy/max_weight': weight_norms['policy_max_weight'],
                    'policy/min_weight': weight_norms['policy_min_weight'],
                }, step=episode)
            except Exception as e:
                print(f"[wandb] Logging failed: {e}")
            
            log_into_wandb=False
        
        if episode % config['save_freq'] == 0 and episode > 0: # Avoid saving at episode 0 before any training
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'dqn_{run_name_suffix}_ep{episode}.pth')
            # Save the policy net of the relevant agent
            if exploration_mode == 'epsilon':
                torch.save(agent.policy_net.state_dict(), checkpoint_path)
            elif exploration_mode != 'epsilon': # RND mode
                # Decide which agent to save, e.g., exploitation agent for final policy
                torch.save(exploitation_agent.policy_net.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path} at episode {episode}")

            # Save a generic latest as well
            latest_checkpoint_path = os.path.join(config['checkpoint_dir'], f'dqn_{run_name_suffix}_latest.pth')
            if exploration_mode == 'epsilon':
                torch.save(agent.policy_net.state_dict(), latest_checkpoint_path)
            elif exploration_mode != 'epsilon':
                torch.save(exploitation_agent.policy_net.state_dict(), latest_checkpoint_path)
            # print(f"Updated latest checkpoint to {latest_checkpoint_path}")

    eval_env.close()
    final_checkpoint_path = os.path.join(config['checkpoint_dir'], f'dqn_{run_name_suffix}_final_ep{config["max_episodes"]}.pth')
    
    # Save the final model based on exploration mode
    final_agent_to_save = None
    if exploration_mode == 'epsilon':
        final_agent_to_save = agent
    elif exploration_mode != 'epsilon': # RND mode
        final_agent_to_save = exploitation_agent # Typically save the exploitation policy

    if final_agent_to_save is not None:
        torch.save(final_agent_to_save.policy_net.state_dict(), final_checkpoint_path)
        param_count = sum(p.numel() for p in final_agent_to_save.policy_net.parameters())
        param_sum = sum(p.sum().item() for p in final_agent_to_save.policy_net.parameters() if p.numel() > 0)
        print(f"\nSaved final model to {final_checkpoint_path}")
        print(f"Parameter count: {param_count:,}")
        print(f"Parameter sum: {param_sum:.2f}")
    else:
        print("\nNo final agent model to save (check exploration_mode logic).")

    print("Training finished.")

if __name__ == '__main__':
    main() 