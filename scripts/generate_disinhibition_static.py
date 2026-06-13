#!/usr/bin/env python3
"""
Generate a static figure matching the thesis disinhibition figure.
Uses entity collection detector to find exact collection moments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import imageio

from src.utils.helpers import (
    load_interpretable_model,
    ModelActivations,
    get_device,
    generate_action,
    observation_to_rgb,
)
from src.utils import heist
from src.utils.entity_collection_detector import get_entity_counts, detect_collections


def find_full_key_seed(model, channel=18, max_seeds=2000):
    """Find a seed where the maze has all 3 keys + locks and the agent completes it."""
    device = get_device()
    model_activations = ModelActivations(model)

    best_seed = None
    best_range = 0

    for seed in range(max_seeds):
        venv = heist.create_venv(num=1, start_level=seed, num_levels=1)
        obs_list = venv.reset()
        obs = obs_list[0] if isinstance(obs_list, (list, np.ndarray)) and len(obs_list.shape) > 3 else obs_list

        # Check initial state for all keys
        state = heist.state_from_venv(venv, 0)
        counts = get_entity_counts(state)

        # We want at least 2 keys present, ideally 3 (count of 2 means on board + in HUD)
        num_keys = sum(1 for k in [4, 5, 6] if counts.get(k, 0) >= 2)
        if num_keys < 2:
            venv.close() if hasattr(venv, 'close') else None
            continue

        # Check initial activation
        obs_single = obs[0] if obs.ndim == 4 else obs
        if obs_single.ndim == 3 and obs_single.shape[0] == 3:
            obs_single = np.transpose(obs_single, (1, 2, 0))
        obs_rgb = observation_to_rgb(obs_single)
        obs_tensor = torch.tensor(obs_rgb.copy(), dtype=torch.float32).unsqueeze(0).to(device)
        _, acts = model_activations.run_with_cache(obs_tensor, ['conv4a'])
        initial_mean = acts['conv4a'].squeeze(0)[channel].cpu().numpy().mean()

        # Run episode to see if agent completes it
        total_reward = 0
        final_mean = initial_mean
        for step in range(400):
            action = generate_action(model, obs)
            obs, reward, done, info = venv.step(action)
            obs = obs[0] if isinstance(obs, np.ndarray) and obs.ndim == 4 else obs
            total_reward += reward
            if step == 399 or (isinstance(done, np.ndarray) and done[0]) or done:
                obs_s = obs[0] if obs.ndim == 4 else obs
                if obs_s.ndim == 3 and obs_s.shape[0] == 3:
                    obs_s = np.transpose(obs_s, (1, 2, 0))
                obs_rgb = observation_to_rgb(obs_s)
                obs_tensor = torch.tensor(obs_rgb.copy(), dtype=torch.float32).unsqueeze(0).to(device)
                _, acts = model_activations.run_with_cache(obs_tensor, ['conv4a'])
                final_mean = acts['conv4a'].squeeze(0)[channel].cpu().numpy().mean()
                break

        venv.close() if hasattr(venv, 'close') else None

        act_range = final_mean - initial_mean
        if total_reward >= 10.0 and act_range > best_range:
            best_range = act_range
            best_seed = seed
            print(f"  Seed {seed}: 3 keys, reward={total_reward}, initial_μ={initial_mean:.1f}, range={act_range:.1f}")

        if seed % 100 == 0:
            print(f"  Tested {seed} seeds... (best: seed={best_seed}, range={best_range:.1f})")

        if best_range > 8.0:
            break

    print(f"  Best seed: {best_seed} (range: {best_range:.1f})")
    return best_seed or 0


def run_episode_with_entity_tracking(model, channel=18, max_steps=400, seed=0):
    """Run episode using venv + entity detector, capture activations at every step."""
    device = get_device()
    model_activations = ModelActivations(model)

    venv = heist.create_venv(num=1, start_level=seed, num_levels=1)
    obs = venv.reset()
    obs = obs[0] if isinstance(obs, np.ndarray) and obs.ndim == 4 else obs

    # Initial state
    state = heist.state_from_venv(venv, 0)
    prev_counts = get_entity_counts(state)
    gem_exists = state.count_entities(9) > 0 or prev_counts.get(3, 0) > 0  # gem entity type

    print(f"  Initial entity counts: {prev_counts}")

    frames = []
    collection_events = []  # (step, entity_name)

    for step in range(max_steps):
        # Raw obs from venv is (N, C, H, W) or (C, H, W)
        obs_single = obs[0] if obs.ndim == 4 else obs

        # For display: transpose CHW -> HWC
        if obs_single.ndim == 3 and obs_single.shape[0] == 3:
            obs_hwc = np.transpose(obs_single, (1, 2, 0))
        else:
            obs_hwc = obs_single
        obs_rgb = observation_to_rgb(obs_hwc)

        # For model: keep CHW format, pass through model for activations
        obs_tensor = torch.tensor(obs_single.copy(), dtype=torch.float32).unsqueeze(0).to(device)
        _, activations = model_activations.run_with_cache(obs_tensor, ['conv4a'])
        conv4a_act = activations['conv4a'].squeeze(0).cpu().numpy()
        channel_act = conv4a_act[channel]
        mean_act = channel_act.mean()

        frames.append({
            'obs': obs_rgb,
            'activation_map': channel_act,
            'mean_activation': mean_act,
            'step': step,
        })

        # Step: pass raw obs to generate_action (it handles format internally)
        action = generate_action(model, obs)
        obs, reward, done, info = venv.step(action)
        obs = obs[0] if isinstance(obs, np.ndarray) and obs.ndim == 4 else obs

        # Check for collections
        done_flag = done[0] if isinstance(done, np.ndarray) else done
        state = heist.state_from_venv(venv, 0)
        current_counts, collected = detect_collections(
            state, prev_counts, episode_done=bool(done_flag), initial_gem_exists=gem_exists
        )
        prev_counts = current_counts

        if collected:
            for entity_name in collected:
                collection_events.append((step, entity_name))
                print(f"  Step {step}: Collected {entity_name} (μ={mean_act:.1f})")

        if done_flag:
            print(f"  Episode ended at step {step}")
            break

    venv.close() if hasattr(venv, 'close') else None
    return frames, collection_events


def make_blue_overlay(obs_rgb, activation_map, vmin=-15):
    """Blue overlay: nearly opaque where strongly negative, transparent where near zero."""
    obs_float = obs_rgb.astype(np.float32) / 255.0
    act_resized = np.array(Image.fromarray(activation_map.astype(np.float32)).resize(
        (obs_float.shape[1], obs_float.shape[0]), Image.NEAREST
    ))

    negative_magnitude = np.clip(-act_resized, 0, None)
    max_magnitude = abs(vmin) if abs(vmin) > 0 else 1
    alpha = np.clip(negative_magnitude / max_magnitude, 0, 1)

    blue_color = np.array([0.02, 0.02, 0.35])
    overlay = obs_float.copy()
    for c in range(3):
        overlay[:, :, c] = obs_float[:, :, c] * (1 - alpha * 0.95) + blue_color[c] * (alpha * 0.95)

    return np.clip(overlay, 0, 1)


def make_static_figure(frames, snapshot_indices, snapshot_labels, channel=18, output_path="disinhibition_static.png"):
    """Create thesis-style static figure."""
    n_cols = len(snapshot_indices)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8,
        'mathtext.default': 'regular',
    })

    # 4 rows x 4 cols: each snapshot gets 2 columns (overlay + heatmap), 4 snapshots per row pair
    # Actually simpler: n_rows = n_snapshots, 2 cols (overlay, heatmap)
    n_rows = n_cols  # n_cols is actually number of snapshots
    fig, axes = plt.subplots(n_rows, 2, figsize=(4, 1.2 * n_rows),
                             gridspec_kw={'hspace': 0.06, 'wspace': 0.04})
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    all_acts = [frames[i]['activation_map'] for i in snapshot_indices]
    vmin = min(a.min() for a in all_acts)
    vmax = max(max(a.max() for a in all_acts), 0)

    cmap = 'RdBu_r'

    for row, (frame_idx, label) in enumerate(zip(snapshot_indices, snapshot_labels)):
        frame = frames[frame_idx]
        obs = frame['obs']
        act_map = frame['activation_map']
        mean_act = frame['mean_activation']

        overlay = make_blue_overlay(obs, act_map, vmin=vmin)
        axes[row, 0].imshow(overlay)
        axes[row, 0].set_ylabel(label, fontsize=6, rotation=0, ha='right', labelpad=8)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        im = axes[row, 1].imshow(act_map, cmap=cmap, vmin=vmin, vmax=0, interpolation='nearest')
        axes[row, 1].set_ylabel(f'$\\mu$={mean_act:.1f}', fontsize=6, rotation=0, ha='left', labelpad=20)
        axes[row, 1].yaxis.set_label_position('right')
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

    axes[0, 0].set_title('Observation', fontsize=7, pad=3)
    axes[0, 1].set_title(f'Conv4a Ch.{channel}', fontsize=7, pad=3)

    cbar = fig.colorbar(im, ax=axes[:, 1].tolist(), fraction=0.03, pad=0.25, shrink=0.7)
    cbar.set_label('Pre-ReLU', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.suptitle('Progressive Disinhibition in Conv4a', fontsize=8, y=1.01)
    plt.subplots_adjust(top=0.96, bottom=0.005, left=0.15, right=0.72, hspace=0.06)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved to {output_path}")


def render_gif_frame(frame, collection_events, channel, vmin, vmax):
    """Render a single GIF frame: observation with overlay on left, activation map on right."""
    obs = frame['obs']
    act_map = frame['activation_map']
    mean_act = frame['mean_activation']
    step = frame['step']

    # Figure out what's been collected so far
    collected_so_far = [name for s, name in collection_events if s < step]
    status = ", ".join(collected_so_far[-2:]) if collected_so_far else "Start"

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'mathtext.default': 'regular',
    })

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    overlay = make_blue_overlay(obs, act_map, vmin=vmin)
    axes[0].imshow(overlay)
    axes[0].set_title(f"Step {step} — {status}", fontsize=10)
    axes[0].axis('off')

    im = axes[1].imshow(act_map, cmap='RdBu_r', vmin=vmin, vmax=0, interpolation='nearest')
    axes[1].set_title(f'Conv4a Ch.{channel} — $\\mu$ = {mean_act:.1f}', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Pre-ReLU')

    plt.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return img


def make_gif(frames, collection_events, channel=18, output_path="disinhibition.gif", fps=8):
    """Generate a GIF from all frames."""
    all_acts = [f['activation_map'] for f in frames]
    vmin = min(a.min() for a in all_acts)
    vmax = max(a.max() for a in all_acts)

    print(f"Rendering {len(frames)} GIF frames (vmin={vmin:.1f}, vmax={vmax:.1f})...")
    rendered = []
    for i, frame in enumerate(frames):
        if i % 20 == 0:
            print(f"  Frame {i}/{len(frames)}")
        rendered.append(render_gif_frame(frame, collection_events, channel, vmin, vmax))

    print(f"Saving GIF to {output_path} ({fps} fps)...")
    imageio.mimsave(output_path, rendered, fps=fps, loop=0)
    print(f"Done! {len(rendered)} frames, {len(rendered)/fps:.1f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="base_models/full_run/model_35001.0.pt")
    parser.add_argument("--channel", type=int, default=18)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", default="disinhibition_static.png")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = load_interpretable_model(model_path=args.model_path)
    model.eval()

    seed = args.seed
    if seed == -1:
        print("Searching for a full 3-key maze seed...")
        seed = find_full_key_seed(model, channel=args.channel)

    print(f"\nRunning episode (seed={seed})...")
    frames, collection_events = run_episode_with_entity_tracking(
        model, channel=args.channel, seed=seed
    )
    print(f"Captured {len(frames)} frames, {len(collection_events)} collections")

    # Build snapshots from collection events
    snapshot_indices = [0]
    snapshot_labels = ["Start"]

    for step, entity_name in collection_events:
        snapshot_indices.append(min(step + 1, len(frames) - 1))
        label = entity_name.replace("_", " ").title()
        snapshot_labels.append(f"After {label}")

    print(f"Snapshots: {list(zip(snapshot_labels, snapshot_indices))}")

    make_static_figure(frames, snapshot_indices, snapshot_labels,
                       channel=args.channel, output_path=args.output)

    # Also generate GIF if output ends with .png, make a .gif version
    gif_path = args.output.replace('.png', '.gif')
    if gif_path != args.output:
        make_gif(frames, collection_events, channel=args.channel, output_path=gif_path)


if __name__ == "__main__":
    main()
