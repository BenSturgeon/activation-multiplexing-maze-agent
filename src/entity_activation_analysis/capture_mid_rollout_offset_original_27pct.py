#!/usr/bin/env python3
"""
Original version that produced the 27% red-first result.
This replicates the exact code from the conversation history.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import random

from src.utils.helpers import (
    load_interpretable_model,
    ModelActivations,
    get_device,
    generate_action,
    observation_to_rgb
)
from src.utils.create_intervention_mazes import create_cross_maze
from src.utils.entity_collection_detector import get_entity_counts, detect_collections
from src.utils import heist

def capture_red_targeting_pattern(model, model_activations, seed):
    """
    Run a rollout and capture activations when the agent is targeting red key
    (after collecting blue and green keys).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_list, venv = create_cross_maze(include_locks=False)
    obs = obs_list[0]

    state = heist.state_from_venv(venv, 0)
    entity_counts = get_entity_counts(state)

    blue_collected = False
    green_collected = False
    red_collected = False
    red_targeting_activations = []
    collections = []

    for step in range(100):
        # Get current activations
        if len(obs.shape) == 4:
            obs_rgb = observation_to_rgb(obs[0])
        else:
            obs_rgb = observation_to_rgb(obs)

        obs_tensor = torch.tensor(obs_rgb, dtype=torch.float32).unsqueeze(0).to(get_device())
        _, activations = model_activations.run_with_cache(obs_tensor, ['conv4a'])

        if 'conv4a' in activations:
            feat = activations['conv4a']
            pooled = torch.mean(feat, dim=(1, 2)).cpu().numpy()

            # Capture activations when targeting RED
            # After blue and green collected, before red
            if blue_collected and green_collected and not red_collected:
                red_targeting_activations.append(pooled)

        # Generate action and step
        action = generate_action(model, obs)
        obs, reward, done, info = venv.step(action)
        obs = obs[0]

        # Check for collections
        state = heist.state_from_venv(venv, 0)
        entity_counts, collected_this_step = detect_collections(state, entity_counts)

        if collected_this_step:
            for item in collected_this_step:
                collections.append(item)

                if 'blue_key' in item:
                    blue_collected = True
                elif 'green_key' in item:
                    green_collected = True
                elif 'red_key' in item:
                    red_collected = True

        if done[0]:
            break

    venv.close()

    if len(red_targeting_activations) > 0:
        # Average the activations from when agent was targeting red
        avg_red_targeting = np.mean(red_targeting_activations, axis=0)
        return avg_red_targeting, collections
    else:
        return None, collections

def get_initial_activation(model, model_activations, seed):
    """Get the initial activation for a specific seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_list, venv = create_cross_maze(include_locks=False)
    obs = obs_list[0]

    if len(obs.shape) == 4:
        obs_rgb = observation_to_rgb(obs[0])
    else:
        obs_rgb = observation_to_rgb(obs)

    obs_tensor = torch.tensor(obs_rgb, dtype=torch.float32).unsqueeze(0).to(get_device())
    _, activations = model_activations.run_with_cache(obs_tensor, ['conv4a'])

    if 'conv4a' in activations:
        feat = activations['conv4a']
        pooled = torch.mean(feat, dim=(1, 2)).cpu().numpy()

    venv.close()
    return pooled

def run_with_intervention(model, seed, offset=None):
    """Run a single rollout with optional offset intervention."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_list, venv = create_cross_maze(include_locks=False)
    obs = obs_list[0]

    state = heist.state_from_venv(venv, 0)
    entity_counts = get_entity_counts(state)

    hook_handle = None
    if offset is not None:
        def modify_conv4a(module, input, output):
            for ch in range(32):
                output[:, ch] += offset[ch]
            return output
        hook_handle = model.conv4a.register_forward_hook(modify_conv4a)

    first_collected = None

    for step in range(100):
        action = generate_action(model, obs)
        obs, reward, done, info = venv.step(action)
        obs = obs[0]

        state = heist.state_from_venv(venv, 0)
        entity_counts, collected_this_step = detect_collections(state, entity_counts)

        if collected_this_step and first_collected is None:
            first_collected = collected_this_step[0]
            break

        if done[0]:
            break

    if hook_handle is not None:
        hook_handle.remove()

    venv.close()
    return first_collected

def main():
    # Load model
    model_path = "base_models/full_run/model_35001.0.pt"
    device = get_device()
    model = load_interpretable_model(model_path=model_path).to(device)
    model.eval()
    model_activations = ModelActivations(model)

    print("=== STEP 1: Collecting per-seed red-targeting patterns ===")

    # Test seeds - using 200-599 for 400 seed sample
    test_seeds = list(range(200, 600))  # 400 seeds for testing

    # First get baseline behavior for each seed
    print("\nGetting baseline behaviors...")
    baselines = {}
    for i, seed in enumerate(test_seeds):
        baselines[seed] = run_with_intervention(model, seed, offset=None)
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(test_seeds)}")

    # Count baseline behaviors
    blue_count = sum(1 for e in baselines.values() if e and 'blue' in e)
    green_count = sum(1 for e in baselines.values() if e and 'green' in e)
    red_count = sum(1 for e in baselines.values() if e and 'red' in e)

    print(f"\nBaseline behavior ({len(test_seeds)} seeds):")
    print(f"  Blue first: {blue_count} ({blue_count/len(test_seeds)*100:.1f}%)")
    print(f"  Green first: {green_count} ({green_count/len(test_seeds)*100:.1f}%)")
    print(f"  Red first: {red_count} ({red_count/len(test_seeds)*100:.1f}%)")

    print("\n=== STEP 2: Capturing per-seed red-targeting patterns ===")

    # For each seed, capture its red-targeting pattern
    seed_red_patterns = {}
    seed_initial_patterns = {}
    successful_captures = 0

    for i, seed in enumerate(test_seeds):
        # Get initial activation
        seed_initial_patterns[seed] = get_initial_activation(model, model_activations, seed)

        # Try to capture red-targeting pattern
        red_pattern, collections = capture_red_targeting_pattern(model, model_activations, seed)

        if red_pattern is not None and len(collections) >= 3:
            # Check if it actually went blue->green->red
            if ('blue_key' in collections[0] and 'green_key' in collections[1] and
                'red_key' in collections[2]):
                seed_red_patterns[seed] = red_pattern
                successful_captures += 1
                if successful_captures <= 3:  # Print first few for verification
                    print(f"  Seed {seed}: Captured red-targeting pattern (blue->green->red)")

        if i % 50 == 0:
            print(f"  Progress: {i}/{len(test_seeds)}, captured {successful_captures} patterns")

    print(f"\nSuccessfully captured {successful_captures}/{len(test_seeds)} red-targeting patterns")

    if successful_captures == 0:
        print("No red-targeting patterns captured! Cannot proceed.")
        return

    print("\n=== STEP 3: Testing per-seed red-targeting offsets ===")

    # Apply per-seed offsets
    red_first_results = 0
    green_first_results = 0
    blue_first_results = 0
    none_results = 0

    tested_count = 0
    for seed in test_seeds:
        if seed in seed_red_patterns:
            # Calculate per-seed offset (ORIGINAL METHOD: red_targeting - initial)
            offset = seed_red_patterns[seed] - seed_initial_patterns[seed]

            # Test with offset
            new_behavior = run_with_intervention(model, seed, offset=offset)
            tested_count += 1

            if new_behavior:
                if 'red' in new_behavior:
                    red_first_results += 1
                elif 'green' in new_behavior:
                    green_first_results += 1
                elif 'blue' in new_behavior:
                    blue_first_results += 1
            else:
                none_results += 1

            if tested_count <= 5:  # Print first few results
                print(f"  Seed {seed}: {baselines[seed]} -> {new_behavior}")

    print(f"\n=== RESULTS: Per-Seed Red-Targeting Offsets ===")
    print(f"Tested on {tested_count} seeds with captured red patterns:")
    print(f"  Red first: {red_first_results} ({red_first_results/tested_count*100:.1f}%)")
    print(f"  Green first: {green_first_results} ({green_first_results/tested_count*100:.1f}%)")
    print(f"  Blue first: {blue_first_results} ({blue_first_results/tested_count*100:.1f}%)")
    print(f"  None: {none_results}")

    print(f"\nCompare to baseline (same seeds):")
    baseline_tested_blue = sum(1 for s in seed_red_patterns.keys() if baselines.get(s) and 'blue' in baselines[s])
    baseline_tested_green = sum(1 for s in seed_red_patterns.keys() if baselines.get(s) and 'green' in baselines[s])
    baseline_tested_red = sum(1 for s in seed_red_patterns.keys() if baselines.get(s) and 'red' in baselines[s])

    print(f"  Blue first: {baseline_tested_blue/tested_count*100:.1f}% -> {blue_first_results/tested_count*100:.1f}%")
    print(f"  Green first: {baseline_tested_green/tested_count*100:.1f}% -> {green_first_results/tested_count*100:.1f}%")
    print(f"  Red first: {baseline_tested_red/tested_count*100:.1f}% -> {red_first_results/tested_count*100:.1f}%")

if __name__ == "__main__":
    main()
