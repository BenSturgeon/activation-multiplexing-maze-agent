#!/usr/bin/env python3
"""
Analyze the exact offset values used in the remarkable green steering result.
This script replicates the experiment and saves detailed offset information.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import numpy as np
import torch
import random
import json

from src.utils.helpers import (
    load_interpretable_model,
    ModelActivations,
    get_device,
    observation_to_rgb
)
from src.utils.create_intervention_mazes import create_expanded_cross_maze

def get_initial_activation(model, model_activations, seed):
    """Get the initial activation for a specific seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_list, venv = create_expanded_cross_maze(seed=seed)
    obs = obs_list[0]

    if len(obs.shape) == 4:
        obs_rgb = observation_to_rgb(obs[0])
    else:
        obs_rgb = observation_to_rgb(obs)

    obs_tensor = torch.tensor(obs_rgb, dtype=torch.float32).unsqueeze(0).to(get_device())
    _, activations = model_activations.run_with_cache(obs_tensor, ['conv4a'])

    pooled = None
    if 'conv4a' in activations:
        feat = activations['conv4a']
        pooled = torch.mean(feat, dim=(1, 2)).cpu().numpy()

    venv.close()
    return pooled

def main():
    # Load model
    print("Loading model...")
    model_path = "base_models/full_run/model_35001.0.pt"
    device = get_device()
    model = load_interpretable_model(model_path=model_path).to(device)
    model.eval()
    model_activations = ModelActivations(model)

    # The exact seeds from the successful experiment
    test_seeds = [882055386, 2639866, 44278588, 372648555, 952142617,
                  227892002, 629683338, 1063938045, 933726074, 870990467]

    print("\n" + "="*70)
    print("ANALYZING GREEN STEERING OFFSET VALUES")
    print("="*70)

    results = {}

    for seed in test_seeds:
        print(f"\nSeed {seed}:")

        # Get initial activation
        initial_act = get_initial_activation(model, model_activations, seed)

        # Calculate offset (negated initial, as used in experiment)
        offset = -initial_act

        print(f"  Initial activation shape: {initial_act.shape}")
        print(f"  Initial mean: {initial_act.mean():.3f}, std: {initial_act.std():.3f}")
        print(f"  Initial range: [{initial_act.min():.3f}, {initial_act.max():.3f}]")

        print(f"  Offset (negated) mean: {offset.mean():.3f}, std: {offset.std():.3f}")
        print(f"  Offset range: [{offset.min():.3f}, {offset.max():.3f}]")

        # Store results
        results[seed] = {
            'initial_activation': initial_act.tolist(),
            'offset_applied': offset.tolist(),
            'offset_stats': {
                'mean': float(offset.mean()),
                'std': float(offset.std()),
                'min': float(offset.min()),
                'max': float(offset.max())
            }
        }

    # Calculate aggregate statistics
    all_offsets = np.array([results[s]['offset_applied'] for s in test_seeds])
    mean_offset = np.mean(all_offsets, axis=0)

    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    print(f"Average offset across all seeds:")
    print(f"  Mean: {mean_offset.mean():.3f}")
    print(f"  Std: {mean_offset.std():.3f}")
    print(f"  Range: [{mean_offset.min():.3f}, {mean_offset.max():.3f}]")

    # Save to file
    output_file = 'green_steering_offset_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'seeds': test_seeds,
            'per_seed_results': results,
            'mean_offset_vector': mean_offset.tolist(),
            'aggregate_stats': {
                'mean': float(mean_offset.mean()),
                'std': float(mean_offset.std()),
                'min': float(mean_offset.min()),
                'max': float(mean_offset.max())
            }
        }, f, indent=2)

    print(f"\nSaved detailed analysis to {output_file}")

    # Show channel-wise statistics
    print("\n" + "="*70)
    print("CHANNEL-WISE OFFSET VALUES (top 5 strongest)")
    print("="*70)
    channel_indices = np.argsort(np.abs(mean_offset))[::-1][:5]
    for idx in channel_indices:
        print(f"  Channel {idx}: {mean_offset[idx]:.3f}")

if __name__ == "__main__":
    main()