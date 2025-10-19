#!/usr/bin/env python3
"""
Sweep experiment: Test uniform offsets from -4 to +6 in 0.1 increments.
Run 10 trials for each offset value to map activation zones for key preferences.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.helpers import (
    load_interpretable_model,
    get_device,
    generate_action,
    observation_to_rgb,
    ModelActivations
)
from src.utils.create_intervention_mazes import create_expanded_cross_maze, create_cross_maze
from src.utils.entity_collection_detector import get_entity_counts, detect_collections
from src.utils import heist

def run_with_uniform_offset(model, seed, offset_value=0.0, max_steps=200, maze_type='expanded', target_layer='conv4a'):
    """Run rollout with uniform offset applied to all channels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if maze_type == 'standard':
        obs_list, venv = create_cross_maze(include_locks=False)
    else:
        obs_list, venv = create_expanded_cross_maze(seed=seed)
    obs = obs_list[0]

    state = heist.state_from_venv(venv, 0)
    entity_counts = get_entity_counts(state)

    # Setup hook for uniform offset
    hook_handle = None
    if offset_value != 0.0:
        def modify_layer(module, input, output):
            # Add uniform offset to all channels
            output[:, :] += offset_value
            return output
        target = getattr(model, target_layer)
        hook_handle = target.register_forward_hook(modify_layer)

    first_collected = None
    all_collected = []

    for step in range(max_steps):
        action = generate_action(model, obs)
        obs, reward, done, info = venv.step(action)
        obs = obs[0]

        state = heist.state_from_venv(venv, 0)
        current_counts, collected_this_step = detect_collections(
            state,
            entity_counts,
            episode_done=done[0],
            initial_gem_exists=True  # Gem always exists in expanded cross maze
        )
        entity_counts = current_counts

        if collected_this_step:
            for item in collected_this_step:
                all_collected.append(item)
                if first_collected is None:
                    first_collected = item
                    break  # We only need first collection

        if first_collected is not None or done[0]:
            break

    if hook_handle is not None:
        hook_handle.remove()
    venv.close()

    return first_collected

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze_type', type=str, default='expanded', choices=['expanded', 'standard'])
    parser.add_argument('--checkpoint', type=str, default='35001', help='Model checkpoint (e.g., 35001, 40001, 45001, 50001)')
    parser.add_argument('--layer', type=str, default='conv4a', choices=['conv2a', 'conv3a', 'conv4a'], help='Target layer for offset')
    parser.add_argument('--offset_start', type=float, default=-15.0, help='Starting offset value')
    parser.add_argument('--offset_end', type=float, default=15.0, help='Ending offset value')
    parser.add_argument('--offset_step', type=float, default=0.2, help='Offset step size')
    args = parser.parse_args()

    print("=" * 70)
    print(f"OFFSET SWEEP EXPERIMENT - {args.layer.capitalize()} - {args.maze_type.upper()} MAZE - CHECKPOINT {args.checkpoint}")
    print(f"Testing uniform offsets from {args.offset_start} to {args.offset_end} in {args.offset_step} increments")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model_path = f"base_models/full_run/model_{args.checkpoint}.0.pt"
    print(f"Model: {model_path}")
    device = get_device()
    model = load_interpretable_model(model_path=model_path).to(device)
    model.eval()

    # Load viable seeds
    viable_seeds_path = "src/utils/viable_seeds.txt"
    with open(viable_seeds_path, 'r') as f:
        viable_seeds = [int(line.strip()) for line in f if line.strip()]

    # Setup experiment parameters
    offset_start = args.offset_start
    offset_end = args.offset_end
    offset_step = args.offset_step
    n_runs_per_offset = 100

    offsets = np.arange(offset_start, offset_end + offset_step, offset_step)

    # Results storage
    results = {
        'offsets': [],
        'blue_counts': [],
        'green_counts': [],
        'red_counts': [],
        'gem_counts': [],
        'none_counts': []
    }

    print(f"\nRunning {len(offsets)} offset values with {n_runs_per_offset} runs each")
    print(f"Total runs: {len(offsets) * n_runs_per_offset}")

    # Run sweep
    seed_idx = 0
    for offset in tqdm(offsets, desc="Offset sweep"):
        blue_count = 0
        green_count = 0
        red_count = 0
        gem_count = 0
        none_count = 0

        # Run n trials for this offset
        for run in range(n_runs_per_offset):
            seed = viable_seeds[seed_idx % len(viable_seeds)]
            seed_idx += 1

            first = run_with_uniform_offset(model, seed, offset_value=offset, maze_type=args.maze_type, target_layer=args.layer)

            if first == 'blue_key':
                blue_count += 1
            elif first == 'green_key':
                green_count += 1
            elif first == 'red_key':
                red_count += 1
            elif first == 'gem':
                gem_count += 1
            else:
                none_count += 1

        results['offsets'].append(offset)
        results['blue_counts'].append(blue_count)
        results['green_counts'].append(green_count)
        results['red_counts'].append(red_count)
        results['gem_counts'].append(gem_count)
        results['none_counts'].append(none_count)

        # Print all results
        print(f"Offset {offset:+.1f}: B={blue_count}, G={green_count}, R={red_count}, Gem={gem_count}, None={none_count}")

    # Create publication-ready visualization
    print("\nCreating publication-ready visualization...")

    # Set style for publication
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Calculate percentages
    total = n_runs_per_offset
    blue_pct = [b/total*100 for b in results['blue_counts']]
    green_pct = [g/total*100 for g in results['green_counts']]
    red_pct = [r/total*100 for r in results['red_counts']]
    gem_pct = [gm/total*100 for gm in results['gem_counts']]
    none_pct = [n/total*100 for n in results['none_counts']]

    # Define colors (using distinct, colorblind-friendly palette)
    colors = {
        'blue': '#1f77b4',    # Blue key
        'green': '#2ca02c',   # Green key
        'red': '#d62728',     # Red key
        'gem': '#ff7f0e',     # Gem (orange/gold)
        'none': '#7f7f7f'     # None (gray)
    }

    # Plot with smooth lines and markers at peaks
    ax.plot(results['offsets'], blue_pct, color=colors['blue'], linewidth=2.5,
            label='Blue Key', alpha=0.9, zorder=3)
    ax.plot(results['offsets'], green_pct, color=colors['green'], linewidth=2.5,
            label='Green Key', alpha=0.9, zorder=3)
    ax.plot(results['offsets'], red_pct, color=colors['red'], linewidth=2.5,
            label='Red Key', alpha=0.9, zorder=3)
    ax.plot(results['offsets'], gem_pct, color=colors['gem'], linewidth=2.0,
            label='Gem', alpha=0.7, linestyle='--', zorder=2)
    ax.plot(results['offsets'], none_pct, color=colors['none'], linewidth=2.0,
            label='None (Failed)', alpha=0.7, linestyle=':', zorder=2)

    # Add vertical line at zero (baseline/no intervention)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
               label='No Intervention', zorder=1)

    # Styling
    layer_label = args.layer.capitalize()
    ax.set_xlabel(f'{layer_label} Activation Offset', fontsize=14)
    ax.set_ylabel('First Collection Rate (%)', fontsize=14)

    # Legend with frame
    legend = ax.legend(loc='upper left', fontsize=12, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')

    # Grid and limits
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_ylim([-2, 102])
    ax.set_xlim([offset_start - 0.5, offset_end + 0.5])

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Add subtle background shading for different regions if there are clear transitions
    # This will be added after we identify the regions

    plt.tight_layout()

    # Save both PNG and PDF
    range_suffix = f'{int(offset_start)}to{int(offset_end)}_step{offset_step}' if (offset_start != -15.0 or offset_end != 15.0 or offset_step != 0.2) else 'full_range'
    base_output = f'src/entity_activation_analysis/plots/offset_sweep_{args.layer}_{args.maze_type}_checkpoint_{args.checkpoint}_{range_suffix}'
    output_png = f'{base_output}.png'
    output_pdf = f'{base_output}.pdf'

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')

    print(f"Saved publication-ready plots:")
    print(f"  PNG: {output_png}")
    print(f"  PDF: {output_pdf}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Find peak offsets for each key
    blue_peak_idx = np.argmax(results['blue_counts'])
    green_peak_idx = np.argmax(results['green_counts'])
    red_peak_idx = np.argmax(results['red_counts'])

    print(f"\nPeak offsets:")
    print(f"  Blue:  {results['offsets'][blue_peak_idx]:+.1f} ({results['blue_counts'][blue_peak_idx]}/{n_runs_per_offset} = {results['blue_counts'][blue_peak_idx]/n_runs_per_offset*100:.0f}%)")
    print(f"  Green: {results['offsets'][green_peak_idx]:+.1f} ({results['green_counts'][green_peak_idx]}/{n_runs_per_offset} = {results['green_counts'][green_peak_idx]/n_runs_per_offset*100:.0f}%)")
    print(f"  Red:   {results['offsets'][red_peak_idx]:+.1f} ({results['red_counts'][red_peak_idx]}/{n_runs_per_offset} = {results['red_counts'][red_peak_idx]/n_runs_per_offset*100:.0f}%)")

    # Find transition points (where preference switches)
    print(f"\nKey transition zones:")
    for i in range(1, len(results['offsets'])):
        prev_max = np.argmax([results['blue_counts'][i-1], results['green_counts'][i-1], results['red_counts'][i-1]])
        curr_max = np.argmax([results['blue_counts'][i], results['green_counts'][i], results['red_counts'][i]])
        if prev_max != curr_max:
            keys = ['Blue', 'Green', 'Red']
            print(f"  {keys[prev_max]} → {keys[curr_max]} at offset ≈ {results['offsets'][i]:+.1f}")

    # Save raw data
    print("\nSaving raw data...")
    data_file = f'src/entity_activation_analysis/results/offset_sweep_{args.layer}_{args.maze_type}_checkpoint_{args.checkpoint}_{range_suffix}_data.txt'
    with open(data_file, 'w') as f:
        f.write("Offset\tBlue\tGreen\tRed\tGem\tNone\n")
        for i in range(len(results['offsets'])):
            f.write(f"{results['offsets'][i]:.1f}\t")
            f.write(f"{results['blue_counts'][i]}\t")
            f.write(f"{results['green_counts'][i]}\t")
            f.write(f"{results['red_counts'][i]}\t")
            f.write(f"{results['gem_counts'][i]}\t")
            f.write(f"{results['none_counts'][i]}\n")
    print(f"Saved raw data to {data_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()