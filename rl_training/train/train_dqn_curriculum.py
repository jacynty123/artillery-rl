import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
from typing import Tuple, Dict, List
import time
from pathlib import Path

try:
    from ..environment import ArtilleryFiringEnv, Action
    from ..curriculum.curriculum_scenarios import CurriculumScenarios
    from ..infrastructure.training_config import TrainingConstants
    from ..agents.dqn_components import DQN, ReplayBuffer
    from ..utils.evaluation_utils import evaluate_on_scenario, evaluate_curriculum, check_phase_completion
    from ..utils.training_monitor import TrainingMonitor
except ImportError:
    # Fallback for standalone execution
    import sys
    import os

    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    # Import from the rl_training package
    from rl_training.environment import ArtilleryFiringEnv, Action
    from rl_training.curriculum.curriculum_scenarios import CurriculumScenarios
    from rl_training.infrastructure.training_config import TrainingConstants
    from rl_training.agents.dqn_components import DQN, ReplayBuffer
    from rl_training.utils.evaluation_utils import evaluate_on_scenario, evaluate_curriculum, check_phase_completion
    from rl_training.utils.training_monitor import TrainingMonitor


def select_action(state, q_network, epsilon, action_dim, device):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()


def train_curriculum_phase(
    phase: int,
    total_timesteps: int = 10000,
    eval_interval: int = TrainingConstants.EVAL_INTERVAL,
    learning_rate: float = TrainingConstants.LEARNING_RATE,
    gamma: float = TrainingConstants.GAMMA,
    epsilon_start: float = TrainingConstants.EPSILON_START,
    epsilon_end: float = TrainingConstants.EPSILON_END,
    epsilon_decay_steps: int = TrainingConstants.EPSILON_DECAY_STEPS,
    batch_size: int = TrainingConstants.BATCH_SIZE,
    buffer_size: int = TrainingConstants.BUFFER_SIZE,
    target_update_freq: int = TrainingConstants.TARGET_UPDATE_FREQ,
    learning_starts: int = TrainingConstants.LEARNING_STARTS,
    device: str = "cpu",
    load_from: str = None,
    firing_reward_high: float = 100.0,
    time_penalty_factor: float = -100.0,
    hold_penalty_high: float = -10.0,
    # Additional configurable reward parameters
    reward_excellent: float = 100.0,
    reward_good: float = 80.0,
    reward_minimum: float = 60.0,
    reward_fair: float = 40.0,
    reward_poor: float = 20.0,
    reward_failure: float = -30.0,
    hold_penalty_base: float = -10.0,
    opportunity_cost_excellent: float = -30.0,
    opportunity_cost_good: float = -15.0,
    opportunity_cost_minimum: float = -8.0
):
    """Train DQN on a curriculum phase."""
    device = torch.device(device)
    
    # Initialize curriculum and environment
    curriculum = CurriculumScenarios(phase=phase)
    # curriculum.scenarios = curriculum.scenarios[:2]  # Removed scenario limiting for full generalization
    curriculum.print_curriculum_info()
    
    # Initialize training monitor
    monitor = TrainingMonitor()
    
    # Create environment without fixed max_episode_steps to enable dynamic step calculation
    env = ArtilleryFiringEnv(
        firing_reward_high=firing_reward_high,
        time_penalty_factor=time_penalty_factor,
        hold_penalty_high=hold_penalty_high,
        reward_excellent=reward_excellent,
        reward_good=reward_good,
        reward_minimum=reward_minimum,
        reward_fair=reward_fair,
        reward_poor=reward_poor,
        reward_failure=reward_failure,
        hold_penalty_base=hold_penalty_base,
        opportunity_cost_excellent=opportunity_cost_excellent,
        opportunity_cost_good=opportunity_cost_good,
        opportunity_cost_minimum=opportunity_cost_minimum
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize networks
    q_network = DQN(state_dim, action_dim).to(device)
    target_network = DQN(state_dim, action_dim).to(device)
    
    # Load previous phase if available
    if load_from and Path(load_from).exists():
        print(f"\nLoading weights from {load_from}")
        q_network.load_state_dict(torch.load(load_from))
    
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    print(f"\nTraining Phase {phase}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Epsilon: {epsilon_start} → {epsilon_end} over {epsilon_decay_steps} steps")
    print(f"Evaluation every {eval_interval} episodes")
    print(f"Starting training...\n")
    
    # Training loop
    scenario = curriculum.get_next_scenario()
    state, _ = env.reset(scenario_override=scenario)
    
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    episode_rewards = []
    
    # Per-scenario tracking
    scenario_stats = defaultdict(lambda: {'rewards': [], 'hps': [], 'steps': []})
    
    start_time = time.time()
    
    final_results = None
    for step in range(total_timesteps):
        # Epsilon decay
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * step / epsilon_decay_steps)

        # Select and execute action
        action = select_action(state, q_network, epsilon, action_dim, device)
        next_state, reward, terminated, truncated, info = env.step(Action(action))
        done = terminated or truncated

        # Log per-step details
        # ...existing code...

        # Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward
        episode_length += 1

        # Training step
        if len(replay_buffer) >= learning_starts and step % 4 == 0:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states_t = torch.FloatTensor(states).to(device)
            actions_t = torch.LongTensor(actions).to(device)
            rewards_t = torch.FloatTensor(rewards).to(device)
            next_states_t = torch.FloatTensor(next_states).to(device)
            dones_t = torch.FloatTensor(dones).to(device)

            # Q-learning update
            current_q = q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_network(next_states_t).max(1)[0]
                target_q = rewards_t + gamma * next_q * (1 - dones_t)

            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Target network update
        if step % target_update_freq == 0 and step > 0:
            target_network.load_state_dict(q_network.state_dict())

        # Episode end
        if done:
            final_hp = info.get('final_hp', 0.0)
            episode_rewards.append(episode_reward)
            # Track per-scenario stats
            scenario_name = scenario.name
            scenario_stats[scenario_name]['rewards'].append(episode_reward)
            scenario_stats[scenario_name]['hps'].append(final_hp)
            scenario_stats[scenario_name]['steps'].append(episode_length)
            
            # Record episode in monitor
            fired = terminated and not truncated  # Fire terminates, hold may timeout
            monitor.record_episode(scenario_name, episode_reward, final_hp, episode_length, fired, phase)
            # ...existing code...
            episode_count += 1
            # Progress reporting
            if episode_count % 20 == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"Step {step:5d} | Ep {episode_count:4d} | ε={epsilon:.3f} | "
                      f"Reward={avg_reward:6.1f} | Len={episode_length:2d} | Time={elapsed:.0f}s")
                
                # Monitor summary every 100 episodes
                if episode_count % 100 == 0:
                    summary = monitor.get_recent_summary(window=100)
                    print(f"Monitor: Avg Reward={summary['avg_reward']:.1f}, Avg HP={summary['avg_hp']:.3f}, Avg Steps={summary['avg_steps']:.1f}")
            # Curriculum evaluation
            if episode_count % eval_interval == 0:
                eval_results = evaluate_curriculum(env, q_network, curriculum, device, verbose=True)
                final_results = eval_results
                # Check if phase complete
                if check_phase_completion(curriculum, eval_results):
                    print(f"\n✓ Phase {phase} mastered after {episode_count} episodes!")
                    break
            # Reset for next episode
            scenario = curriculum.get_next_scenario()
            state, _ = env.reset(scenario_override=scenario)
            episode_reward = 0
            episode_length = 0
        else:
            state = next_state
    
    # Save model to checkpoints directory
    save_dir = Path(__file__).parent.parent / "models" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"dqn_curriculum_phase{phase}.pth"
    torch.save(q_network.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    
    # If phase was never mastered, run final evaluation
    if final_results is None:
        final_results = evaluate_curriculum(env, q_network, curriculum, device, verbose=True)
    return q_network, final_results


if __name__ == "__main__":
    print("=" * 80)
    print("DQN CURRICULUM LEARNING")
    print("=" * 80)
    
    import argparse
    import json

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="DQN Curriculum Training")
        parser.add_argument("--start_phase", type=int, default=1, help="First curriculum phase to train (1-4)")
        parser.add_argument("--end_phase", type=int, default=3, help="Last curriculum phase to train (1-4)")
        parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps per phase")
        parser.add_argument("--config", type=str, help="JSON config file for parameter overrides")
        args = parser.parse_args()

        # Load config if provided
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)

        prev_checkpoint = None
        for phase in range(args.start_phase, args.end_phase + 1):
            print(f"\n\nSTARTING PHASE {phase}: Curriculum Training")
            print("=" * 80)
            q_network, results = train_curriculum_phase(
                phase=phase,
                total_timesteps=args.timesteps,
                eval_interval=config.get("eval_interval", 100),
                epsilon_decay_steps=config.get("epsilon_decay_steps", 2000),
                firing_reward_high=config.get("firing_reward_high", 100.0),
                time_penalty_factor=config.get("time_penalty_factor", -100.0),
                hold_penalty_high=config.get("hold_penalty_high", -10.0),
                reward_excellent=config.get("reward_excellent", 100.0),
                reward_good=config.get("reward_good", 80.0),
                reward_minimum=config.get("reward_minimum", 60.0),
                reward_fair=config.get("reward_fair", 40.0),
                reward_poor=config.get("reward_poor", 20.0),
                reward_failure=config.get("reward_failure", -30.0),
                hold_penalty_base=config.get("hold_penalty_base", -10.0),
                opportunity_cost_excellent=config.get("opportunity_cost_excellent", -30.0),
                opportunity_cost_good=config.get("opportunity_cost_good", -15.0),
                opportunity_cost_minimum=config.get("opportunity_cost_minimum", -8.0),
                load_from=prev_checkpoint
            )
            prev_checkpoint = str(Path(__file__).parent.parent / "models" / "checkpoints" / f"dqn_curriculum_phase{phase}.pth")

        print("\n" + "=" * 80)
        print("CURRICULUM LEARNING COMPLETE!")
        print("=" * 80)
