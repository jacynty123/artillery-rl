"""
Evaluation Utilities for DQN Curriculum Training

Contains functions for evaluating the policy on scenarios and curriculum.
"""

import numpy as np
import torch
from typing import Dict, List

from ..infrastructure.training_config import TrainingConstants


def rollout(env, q_network, scenario, device):
    """Run one greedy episode and return per-episode stats dict."""
    state, _ = env.reset(scenario_override=scenario)
    episode_reward = 0.0
    done = False
    step_count = 0
    fired_at_step = None
    info = {}

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = q_network(state_tensor).argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        if action == 1 and fired_at_step is None:  # Action.FIRE == 1
            fired_at_step = step_count + 1  # 1-based step
        done = terminated or truncated
        episode_reward += reward
        step_count += 1
        state = next_state

    return {
        'reward': episode_reward,
        'hp': info.get('hit_probability', 0.0),
        'steps': step_count,
        'fired_at_step': fired_at_step,
    }


def evaluate_on_scenario(env, q_network, scenario, device, n_episodes=TrainingConstants.EVAL_EPISODES_PER_SCENARIO):
    """Evaluate policy on a specific scenario."""
    rewards = []
    hps = []
    steps_list = []
    firing_ranges = []
    firing_steps = []

    for _ in range(n_episodes):
        ep = rollout(env, q_network, scenario, device)
        initial_range = scenario.range_m
        fired_at_step = ep['fired_at_step']
        step_count = ep['steps']
        final_hp = ep['hp']

        if TrainingConstants.DEBUG:
            print(f"Scenario: {scenario.name}, Fired at step: {fired_at_step}, Final HP: {final_hp:.3f}")

        elapsed_time = step_count * (scenario.tracking_duration / env.max_episode_steps)
        firing_range = initial_range + scenario.target_vx * elapsed_time

        rewards.append(ep['reward'])
        hps.append(final_hp)
        steps_list.append(step_count)
        firing_ranges.append(firing_range)
        firing_steps.append(fired_at_step if fired_at_step is not None else step_count)

    return {
        'reward': np.mean(rewards),
        'hp': np.mean(hps),
        'steps': np.mean(steps_list),
        'firing_range': np.mean(firing_ranges),
        'hp_std': np.std(hps),
        'range_safe': np.mean(firing_ranges) > 200.0,
        'firing_step': np.mean(firing_steps),
        'firing_step_std': np.std(firing_steps)
    }


def evaluate_curriculum(env, q_network, curriculum, device, verbose=True):
    """Evaluate policy on all curriculum scenarios."""
    all_scenarios = curriculum.get_all_scenarios()
    results = []

    if verbose:
        print("\n" + "=" * 80)
        print("CURRICULUM EVALUATION")
        print("=" * 80)

    for i, scenario in enumerate(all_scenarios):
        stats = evaluate_on_scenario(env, q_network, scenario, device, n_episodes=TrainingConstants.EVAL_EPISODES_PER_SCENARIO)
        results.append(stats)

        if verbose:
            difficulty = curriculum.difficulty[scenario.name]
            safety = "[OK]" if stats['range_safe'] else "[UNSAFE]"
            print(f"{i+1}. {scenario.name:30s} [{difficulty:4s}] | "
                f"HP={stats['hp']:.3f}+/-{stats['hp_std']:.3f} | "
                f"Steps={stats['steps']:.1f} | "
                f"FiringStep={stats['firing_step']:.1f}+/-{stats['firing_step_std']:.1f} | "
                f"Range={stats['firing_range']:.0f}m {safety}")

    # Summary statistics
    avg_hp = np.mean([r['hp'] for r in results])
    min_hp = np.min([r['hp'] for r in results])
    all_safe = all(r['range_safe'] for r in results)

    if verbose:
        print("-" * 80)
        print(f"Average HP: {avg_hp:.3f}")
        print(f"Min HP: {min_hp:.3f}")
        print(f"All ranges safe: {all_safe}")
        print("=" * 80)

    return {
        'results': results,
        'avg_hp': avg_hp,
        'min_hp': min_hp,
        'all_safe': all_safe
    }


def check_phase_completion(curriculum, eval_results):
    """Check if current phase requirements are met."""
    requirements = curriculum.get_phase_requirements()

    avg_hp = eval_results['avg_hp']
    min_hp = eval_results['min_hp']
    all_safe = eval_results['all_safe']

    # Get average firing step across all scenarios
    avg_firing_step = np.mean([r['firing_step'] for r in eval_results['results']])
    max_steps = TrainingConstants.MAX_EPISODE_STEPS  # Assuming 50 max steps

    # TIMING EFFICIENCY REQUIREMENT
    timing_target = max_steps * 0.7  # Should fire by 70% of time (step 35)
    timing_efficient = avg_firing_step <= timing_target

    meets_avg = avg_hp >= requirements['min_avg_hp']
    meets_min = min_hp >= requirements['min_hp_per_scenario']

    print("\n" + "=" * 80)
    print("PHASE COMPLETION CHECK")
    print("=" * 80)
    print(f"Phase {curriculum.phase}: {requirements['description']}")
    print(f"  Average HP >= {requirements['min_avg_hp']:.2f}: {avg_hp:.3f} {'[OK]' if meets_avg else '[FAIL]'}")
    print(f"  Min HP >= {requirements['min_hp_per_scenario']:.2f}: {min_hp:.3f} {'[OK]' if meets_min else '[FAIL]'}")
    print(f"  All ranges safe: {'[OK]' if all_safe else '[FAIL]'}")
    print(f"  Timing efficiency (avg step <= {timing_target:.0f}): {avg_firing_step:.1f} {'[OK]' if timing_efficient else '[FAIL]'}")

    complete = meets_avg and meets_min and all_safe and timing_efficient

    if complete:
        print(f"\n[OK] PHASE {curriculum.phase} COMPLETE!")
    else:
        if not timing_efficient:
            print(f"\n[WARN] Timing too slow - fire earlier! (current: {avg_firing_step:.1f}, target: <={timing_target:.0f})")
        else:
            print(f"\n[WARN] Phase {curriculum.phase} not yet complete - continue training")

    print("=" * 80)

    return complete