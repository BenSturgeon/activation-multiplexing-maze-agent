#!/usr/bin/env python3
"""
No-entity (zero target entity) control for the spatial-gating claim.

Examiner question (E2): "was an investigation conducted where there are not any
entities in the maze and did this result in positive mean activation trajectories?"

Spatial-gating prediction: negative conv activations mark regions the agent still
needs to reach. If NO target entity exists, there is nothing to seek, so the
negative-activation signal should largely vanish (trajectories stay non-negative)
relative to the entity-present condition.

Paired design: for each seed we run the SAME maze layout twice --
  (a) standard   : entities present
  (b) no_entity  : gem + keys + locks stripped (heist remove_gem/delete_keys/delete_locks)
and capture conv4a per-channel mean (spatial-pooled) activation at every timestep.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import json
import numpy as np
import torch

from src.entity_activation_analysis.rollout_lib import (
    load_model, make_venv, rollout, CONV4A as LAYER,
)

MAX_STEPS = 100


def run_rollout(model, model_activations, seed, strip):
    """One rollout; return conv4a pooled vectors (T, C). Thin wrapper over the
    shared rollout primitive so behavior is pinned by verify_rollout.py."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    venv, obs = make_venv(seed, strip=strip)
    traj = rollout(model, model_activations, venv, obs, max_steps=MAX_STEPS)
    venv.close()
    return traj  # (T, C)


def summarize(stacked):
    """stacked: (N_total_steps, C). Return examiner-facing metrics."""
    flat = stacked.ravel()
    return {
        "n_step_channel_values": int(flat.size),
        "mean_activation": float(flat.mean()),
        "mean_magnitude": float(np.abs(flat).mean()),
        "frac_negative": float((flat < 0).mean()),
        "min": float(flat.min()),
        "p10": float(np.percentile(flat, 10)),
        "median": float(np.median(flat)),
        "per_channel_mean": stacked.mean(axis=0).tolist(),
        "per_channel_frac_negative": (stacked < 0).mean(axis=0).tolist(),
    }


def per_step_trajectory(trajs, max_steps=MAX_STEPS):
    """trajs: list of (T_i, C) arrays. Return step-aligned mean-over-channels
    trajectory (nanmean across seeds) so conditions can be compared at matched
    timesteps despite ragged episode lengths."""
    grid = np.full((len(trajs), max_steps), np.nan)
    for i, t in enumerate(trajs):
        chan_mean = t.mean(axis=1)  # (T_i,) mean over channels
        n = min(len(chan_mean), max_steps)
        grid[i, :n] = chan_mean[:n]
    with np.errstate(invalid='ignore'):
        step_mean = np.nanmean(grid, axis=0)
        step_std = np.nanstd(grid, axis=0)
        step_n = np.sum(~np.isnan(grid), axis=0)
    return {
        "step_mean": np.where(np.isnan(step_mean), None, step_mean).tolist(),
        "step_std": np.where(np.isnan(step_std), None, step_std).tolist(),
        "step_n": step_n.tolist(),
        "first10_mean": float(np.nanmean(grid[:, :10])),
        "first20_mean": float(np.nanmean(grid[:, :20])),
    }


def main():
    p = argparse.ArgumentParser(description="No-entity control for spatial gating")
    p.add_argument('--num-seeds', type=int, default=30)
    p.add_argument('--seed-start', type=int, default=100)
    p.add_argument('--output-dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), 'results'))
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, model_activations = load_model()

    conditions = {"standard": False, "no_entity": True}
    collected = {k: [] for k in conditions}

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    for seed in seeds:
        for cond, strip in conditions.items():
            traj = run_rollout(model, model_activations, seed, strip)
            if traj.size:
                collected[cond].append(traj)
        print(f"seed {seed}: "
              f"standard {collected['standard'][-1].shape if collected['standard'] else 'NA'}, "
              f"no_entity {collected['no_entity'][-1].shape if collected['no_entity'] else 'NA'}")

    summary = {}
    trajectory = {}
    for cond in conditions:
        stacked = np.concatenate(collected[cond], axis=0)  # (sum T, C)
        summary[cond] = summarize(stacked)
        trajectory[cond] = per_step_trajectory(collected[cond])

    # Headline contrast
    headline = {
        "layer": LAYER,
        "num_seeds": args.num_seeds,
        "standard_mean_activation": summary["standard"]["mean_activation"],
        "no_entity_mean_activation": summary["no_entity"]["mean_activation"],
        "mean_activation_shift_up": (
            summary["no_entity"]["mean_activation"] - summary["standard"]["mean_activation"]
        ),
        "standard_frac_negative": summary["standard"]["frac_negative"],
        "no_entity_frac_negative": summary["no_entity"]["frac_negative"],
        # matched-window (first 20 steps, both conditions have data here)
        "standard_first20_mean": trajectory["standard"]["first20_mean"],
        "no_entity_first20_mean": trajectory["no_entity"]["first20_mean"],
    }

    out = {"headline": headline, "summary": summary,
           "trajectory": trajectory, "seeds": seeds}
    out_path = os.path.join(args.output_dir, "no_entity_control.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("NO-ENTITY CONTROL — conv4a mean-activation trajectories")
    print("=" * 60)
    print(f"standard   : mean={headline['standard_mean_activation']:+.4f}  "
          f"frac_neg={headline['standard_frac_negative']:.3f}  "
          f"first20={headline['standard_first20_mean']:+.4f}")
    print(f"no_entity  : mean={headline['no_entity_mean_activation']:+.4f}  "
          f"frac_neg={headline['no_entity_frac_negative']:.3f}  "
          f"first20={headline['no_entity_first20_mean']:+.4f}")
    print(f"mean shift UP when entities removed (less negative): "
          f"{headline['mean_activation_shift_up']:+.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
