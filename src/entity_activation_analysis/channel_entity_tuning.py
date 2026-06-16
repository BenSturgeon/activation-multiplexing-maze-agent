#!/usr/bin/env python3
"""
In-distribution channel tuning: how does each conv4a channel's mean activation
vary with WHICH entity is the current target (blue key -> green key -> red key
-> gem), during normal rollouts with all entities present?

Answers "does channel c encode entity identity via its activation level?" without
the OOD empty-maze confound. Reports per-channel mean per target phase + which
channels are most entity-tuned (highest inter-phase variance). ch1 highlighted
because the no-entity control flagged it as near-fully entity-dependent.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse, json
import numpy as np
import torch

from src.entity_activation_analysis.rollout_lib import (
    load_model, rollout, CONV4A as LAYER,
)
from src.utils.create_intervention_mazes import create_cross_maze
from src.utils.simple_entity_tracker import EntityTracker
from src.utils import heist

PHASES = ['blue', 'green', 'red', 'gem']
MAX_STEPS = 120


def _current_phase(collected):
    """First uncollected entity in the sequence = the current target phase."""
    for phase in PHASES:
        if not any(phase in name for name in collected):
            return phase
    return 'gem'


def run_rollout(model, model_activations, seed):
    """Return dict phase -> list of conv4a pooled vectors captured while that
    phase's entity was the current target. Uses the shared rollout primitive and
    EntityTracker for collection alignment."""
    np.random.seed(seed); torch.manual_seed(seed)
    obs_list, venv = create_cross_maze(include_locks=False)
    obs = obs_list[0]
    tracker = EntityTracker(heist.state_from_venv(venv, 0))
    buckets = {p: [] for p in PHASES}

    def on_capture(step, pooled, state):
        tracker.update(state, step)  # register collections up to this step
        buckets[_current_phase(tracker.get_collected_entities())].append(pooled)

    rollout(model, model_activations, venv, obs, max_steps=MAX_STEPS,
            on_capture=on_capture)
    venv.close()
    return buckets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-seeds', type=int, default=30)
    p.add_argument('--seed-start', type=int, default=200)
    p.add_argument('--output-dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), 'results'))
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, ma = load_model()

    agg = {p: [] for p in PHASES}
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        b = run_rollout(model, ma, seed)
        for ph in PHASES:
            if b[ph]:
                agg[ph].extend(b[ph])
        print(f"seed {seed}: " + " ".join(f"{ph}={len(b[ph])}" for ph in PHASES))

    # per-channel mean per phase
    n_ch = None
    phase_means = {}
    for ph in PHASES:
        if agg[ph]:
            arr = np.array(agg[ph])  # (N, C)
            phase_means[ph] = arr.mean(axis=0)
            n_ch = arr.shape[1]
    C = n_ch
    # inter-phase variance per channel = how strongly the channel is entity-tuned
    present = [ph for ph in PHASES if ph in phase_means]
    stacked = np.stack([phase_means[ph] for ph in present], axis=0)  # (P, C)
    inter_phase_var = stacked.var(axis=0)  # (C,)
    rank = np.argsort(-inter_phase_var)

    out = {
        "phases_present": present,
        "n_samples_per_phase": {ph: len(agg[ph]) for ph in PHASES},
        "per_channel_mean_by_phase": {ph: phase_means[ph].tolist() for ph in present},
        "inter_phase_variance": inter_phase_var.tolist(),
        "channels_by_entity_tuning": rank.tolist(),
        "ch1_by_phase": {ph: float(phase_means[ph][1]) for ph in present},
    }
    path = os.path.join(args.output_dir, "channel_entity_tuning.json")
    json.dump(out, open(path, 'w'), indent=2)

    print("\n" + "=" * 60)
    print(f"CHANNEL ENTITY TUNING ({LAYER}) — mean activation by current target")
    print("=" * 60)
    print("ch1 across target phases:")
    for ph in present:
        print(f"  {ph:5s}: {phase_means[ph][1]:+.3f}")
    print(f"\nTop 8 most entity-tuned channels (inter-phase variance):")
    for c in rank[:8]:
        vals = " ".join(f"{ph}={phase_means[ph][c]:+.2f}" for ph in present)
        print(f"  ch{c:>2} var={inter_phase_var[c]:.3f}  {vals}")
    print(f"\nch1 rank among 32 by entity-tuning: {list(rank).index(1)+1}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
