#!/usr/bin/env python3
"""
Generate a GIF showing progressive disinhibition in conv4a during a Heist episode.

For each timestep captures:
1. The game observation
2. The pre-ReLU spatial activation map for conv4a channel 18
3. A composite: observation with blue overlay showing negative activation regions

Outputs a GIF with side-by-side observation + activation map per frame.
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
from matplotlib.colors import Normalize
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


def find_good_seed(model, max_seeds=200, min_reward=9.0):
    """Find a seed where the agent successfully completes a multi-key maze."""
    import gym
    device = get_device()

    for seed in range(max_seeds):
        env = gym.make("procgen:procgen-heist-v0", distribution_mode="easy", num_levels=1, start_level=seed)
        obs = env.reset()
        total_reward = 0
        steps = 0

        for step in range(300):
            action = generate_action(model, obs)
            if isinstance(action, np.ndarray):
                action = action.item() if action.size == 1 else action[0]
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps = step
            if done:
                break

        env.close()

        if total_reward >= min_reward and steps > 30:
            print(f"  Seed {seed}: reward={total_reward}, steps={steps} - GOOD")
            return seed
        elif seed % 50 == 0:
            print(f"  Tested {seed} seeds so far...")

    print("No good seed found, using seed 0")
    return 0


def run_episode_capture_activations(model, channel=18, max_steps=300, seed=None):
    """Run an episode and capture observation + conv4a pre-ReLU activations at every step."""

    device = get_device()
    model_activations = ModelActivations(model)

    import gym
    env = gym.make("procgen:procgen-heist-v0", distribution_mode="easy", num_levels=1, start_level=seed or 0)
    obs = env.reset()

    frames = []

    for step in range(max_steps):
        # Get observation as RGB
        obs_rgb = observation_to_rgb(obs)  # (64, 64, 3) uint8

        # Hook conv4a to get pre-ReLU output
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).unsqueeze(0).to(device)
        _, activations = model_activations.run_with_cache(obs_tensor, ['conv4a'])

        # conv4a hook captures Conv2d output = pre-ReLU, shape (1, 32, H, W)
        conv4a_act = activations['conv4a'].squeeze(0).cpu().numpy()  # (32, H, W)
        channel_act = conv4a_act[channel]  # (H, W) - pre-ReLU, has negative values

        mean_act = channel_act.mean()

        frames.append({
            'obs': obs_rgb,
            'activation_map': channel_act,
            'mean_activation': mean_act,
            'step': step,
        })

        # Step environment
        action = generate_action(model, obs)
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
        obs, reward, done, info = env.step(action)

        if done:
            print(f"Episode ended at step {step} (reward collected)")
            break

    env.close()
    return frames


def render_frame(frame_data, vmin=-15, vmax=2):
    """Render a single frame matching the thesis figure style.

    Top row: observation with blue overlay (intensity proportional to negative activation magnitude)
    Bottom row: spatial activation map for conv4a channel 18
    """

    obs = frame_data['obs']
    act_map = frame_data['activation_map']
    mean_act = frame_data['mean_activation']
    step = frame_data['step']

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    # Top: Observation with blue overlay for negative activations
    obs_float = obs.astype(np.float32) / 255.0

    # Resize activation map to observation size (act_map is smaller, e.g. 4x4)
    h, w = act_map.shape
    act_resized = np.array(Image.fromarray(act_map.astype(np.float32)).resize(
        (64, 64), Image.BILINEAR
    ))

    # Blue overlay: intensity proportional to magnitude of negative activations
    # Only show negative regions; positive regions get no overlay
    negative_only = np.clip(-act_resized, 0, None)  # zero out positive, keep magnitude of negative
    max_neg = abs(vmin) if abs(vmin) > 0 else 1
    blue_intensity = np.clip(negative_only / max_neg, 0, 1)  # normalize to [0, 1]

    # Apply blue overlay: blend observation with blue proportional to negative magnitude
    overlay = obs_float.copy()
    alpha = blue_intensity * 0.7  # max 70% overlay
    blue_color = np.array([0.1, 0.2, 0.8])  # blue tint
    for c in range(3):
        overlay[:, :, c] = overlay[:, :, c] * (1 - alpha) + blue_color[c] * alpha

    axes[0].imshow(np.clip(overlay, 0, 1))
    axes[0].set_title(f"Step {step} — Observation + negative activation overlay", fontsize=11)
    axes[0].axis('off')

    # Bottom: Raw spatial activation map
    im = axes[1].imshow(act_map, cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].set_title(f"Conv4a Channel 18 — μ = {mean_act:.1f}", fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Pre-ReLU activation')

    plt.tight_layout()

    # Convert figure to image array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]  # drop alpha channel
    plt.close(fig)

    return img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="base_models/full_run/model_35001.0.pt")
    parser.add_argument("--channel", type=int, default=18)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="disinhibition_gif.gif")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = load_interpretable_model(model_path=args.model_path)
    model.eval()

    seed = args.seed
    if seed == -1:
        print("Searching for a good seed (agent completes multi-key maze)...")
        seed = find_good_seed(model)

    print(f"Running episode (seed={seed}, max_steps={args.max_steps})...")
    frames = run_episode_capture_activations(
        model, channel=args.channel, max_steps=args.max_steps, seed=seed
    )
    print(f"Captured {len(frames)} frames")

    # Find activation range across all frames for consistent colormap
    all_acts = [f['activation_map'] for f in frames]
    vmin = min(a.min() for a in all_acts)
    vmax = max(a.max() for a in all_acts)
    print(f"Activation range: [{vmin:.1f}, {vmax:.1f}]")

    print("Rendering frames...")
    rendered = []
    for i, frame in enumerate(frames):
        if i % 20 == 0:
            print(f"  Frame {i}/{len(frames)}")
        rendered.append(render_frame(frame, vmin=vmin, vmax=vmax))

    print(f"Saving GIF to {args.output} ({args.fps} fps)...")
    imageio.mimsave(args.output, rendered, fps=args.fps, loop=0)
    print(f"Done! {len(rendered)} frames, {len(rendered)/args.fps:.1f}s duration")


if __name__ == "__main__":
    main()
