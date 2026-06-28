#!/usr/bin/env python3
"""
Fig 4.1 rerun with the controls the examiner asked for
(e2-c3-fig41-variance-control-channels):
  - 95% CI bands across seeds (variance shown, not a single rollout)
  - all 32 conv4a channels (grid; top-variance highlighted)
  - a no-entity baseline trajectory (control)

Parallel rollout: the agent walks one T-corridor trajectory; at every step we
swap in each target entity and capture conv4a per-channel mean, isolating the
effect of entity identity. Reuses the validated swap/capture machinery.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.entity_activation_analysis.rollout_lib import load_model, capture_pooled, CONV4A
from src.entity_activation_analysis.t_corridor_experiment import swap_entity_in_state, ENTITY_CODES
from src.utils.create_intervention_mazes import create_t_corridor_maze
from src.utils.helpers import generate_action
from src.utils import heist

MAX_STEPS = 12
TARGETS = [e for e in ['blue_key', 'green_key', 'red_key', 'gem'] if e in ENTITY_CODES]
COLORS = {'blue_key': 'tab:blue', 'green_key': 'tab:green', 'red_key': 'tab:red', 'gem': 'goldenrod'}


COLOR_MAP = {4: 0, 5: 1, 6: 2}  # blue, green, red key themes


def _player_and_target(venv):
    """Current player cell and the target-entity cell (the one non-HUD world entity)."""
    state = heist.state_from_venv(venv, 0)
    player = target = None
    for ent in state.state_vals.get("ents", []):
        t = ent["image_type"].val
        ex, ey = ent["x"].val, ent["y"].val
        if ex is None or ey is None or not (np.isfinite(ex) and np.isfinite(ey)):
            continue
        col, row = int(round(ex - 0.5)), int(round(ey - 0.5))
        if t == 0:
            player = (col, row)
        elif not heist._is_hud_entity(ent) and ex >= 0 and ey >= 0:
            target = (col, row)
    return player, target


def _place_entity(venv, entity_code, target_col, target_row, player):
    """Place ONE entity of entity_code at a FIXED target cell, keep the player where it
    is, return the observation. Works even after the original entity was collected."""
    state = heist.state_from_venv(venv, 0)
    state.remove_all_entities()
    if player is not None:
        state.set_mouse_pos(player[1], player[0])  # (row, col)
    if entity_code == 3:  # gem
        state.set_entity_position(9, None, target_col, target_row)
    else:
        state.set_entity_position(2, COLOR_MAP[entity_code], target_col, target_row)
    venv.env.callmethod("set_state", [state.state_bytes])
    obs = venv.reset()
    return obs[0] if obs.ndim == 4 else obs


def parallel_rollout(model, ma, seed, max_steps=MAX_STEPS):
    np.random.seed(seed); torch.manual_seed(seed)
    _, venv = create_t_corridor_maze(entity_code=ENTITY_CODES['blue_key'], gem_on_left=False)
    # Cache the target cell (arm end) from the initial maze, then jitter the column
    # along the arm per seed (matches the original "entity positions within the arm
    # were randomised for meaningful statistical testing") so the CI bands are real.
    _, target = _player_and_target(venv)
    if target is None:
        venv.close(); return []
    tcol, trow = target
    tcol = tcol - int(np.random.randint(0, 4))  # 0-3 cells toward maze centre along the arm
    per_step = []
    for _ in range(max_steps):
        player, _ = _player_and_target(venv)
        ent = {}
        for name in TARGETS:
            obs = _place_entity(venv, ENTITY_CODES[name], tcol, trow, player)
            ent[name] = capture_pooled(ma, obs, CONV4A)
        per_step.append(ent)
        # Restore base entity (blue key) at the fixed target, then advance the agent.
        base_obs = _place_entity(venv, ENTITY_CODES['blue_key'], tcol, trow, player)
        action = generate_action(model, base_obs)
        _, _, done, _ = venv.step(action)
        if np.array(done).ravel()[0]:
            break
    venv.close()
    return per_step


def no_entity_rollout(model, ma, seed, max_steps=MAX_STEPS):
    np.random.seed(seed); torch.manual_seed(seed)
    _, venv = create_t_corridor_maze(entity_code=ENTITY_CODES['blue_key'], gem_on_left=False)
    st = heist.state_from_venv(venv, 0); st.remove_gem(); st.delete_keys(); st.delete_locks()
    venv.env.callmethod("set_state", [st.state_bytes])
    obs = venv.reset(); obs = obs[0] if obs.ndim == 4 else obs
    traj = []
    for _ in range(max_steps):
        traj.append(capture_pooled(ma, obs, CONV4A))
        action = generate_action(model, obs)
        obs, _, done, _ = venv.step(action); obs = obs[0]
        if np.array(done).ravel()[0]:
            break
    venv.close()
    return traj


def mean_ci(a):  # a: (seeds, T, C) with nans
    m = np.nanmean(a, axis=0); sd = np.nanstd(a, axis=0)
    n = np.sum(~np.isnan(a), axis=0)
    return m, 1.96 * sd / np.sqrt(np.maximum(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-seeds', type=int, default=20)
    ap.add_argument('--seed-start', type=int, default=300)
    ap.add_argument('--output-dir', default='outputs')
    a = ap.parse_args(); os.makedirs(a.output_dir, exist_ok=True)
    model, ma = load_model()

    runs, base = [], []
    for s in range(a.seed_start, a.seed_start + a.num_seeds):
        runs.append(parallel_rollout(model, ma, s))
        base.append(no_entity_rollout(model, ma, s))
        print(f"seed {s}: steps={len(runs[-1])}")

    T = min(MAX_STEPS, min(len(r) for r in runs), min(len(r) for r in base))
    C = 32
    ent = {e: np.full((len(runs), T, C), np.nan) for e in TARGETS}
    for si, r in enumerate(runs):
        for t in range(min(len(r), T)):
            for e in TARGETS:
                if r[t].get(e) is not None:
                    ent[e][si, t, :] = r[t][e]
    barr = np.full((len(base), T, C), np.nan)
    for si, r in enumerate(base):
        for t in range(min(len(r), T)):
            barr[si, t, :] = r[t]

    stats = {e: mean_ci(ent[e]) for e in TARGETS}
    bm, bci = mean_ci(barr)
    chan_means = np.stack([stats[e][0].mean(axis=0) for e in TARGETS], axis=0)
    inter_var = chan_means.var(axis=0)
    order = np.argsort(-inter_var)

    fig, axes = plt.subplots(4, 8, figsize=(26, 12)); axes = axes.ravel()
    x = np.arange(T)
    for i, ch in enumerate(order):
        ax = axes[i]
        for e in TARGETS:
            m, ci = stats[e]
            ax.plot(x, m[:, ch], color=COLORS.get(e, 'gray'), lw=1)
            ax.fill_between(x, m[:, ch] - ci[:, ch], m[:, ch] + ci[:, ch], color=COLORS.get(e, 'gray'), alpha=0.18)
        ax.plot(x, bm[:, ch], 'k--', lw=1, label='no-entity')
        ax.set_title(f'ch{ch} (var={inter_var[ch]:.1f})', fontsize=8)
        ax.tick_params(labelsize=6)
    handles = [plt.Line2D([0], [0], color=COLORS[e], label=e) for e in TARGETS] + \
              [plt.Line2D([0], [0], color='k', ls='--', label='no-entity')]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    fig.suptitle(f'Conv4a parallel-rollout trajectories, mean ±95% CI ({a.num_seeds} seeds), all 32 channels (sorted by inter-entity variance)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(a.output_dir, 'fig41_parallel_rollout_ci.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=120)

    # --- focused figure matching the original thesis Fig 4.1 (channels 27/18/8, stacked) ---
    DISP = {'blue_key': 'blue key', 'gem': 'gem', 'green_key': 'green key', 'red_key': 'red key'}
    ORDER = ['blue_key', 'gem', 'green_key', 'red_key']  # legend order as in original
    orig_channels = [27, 18, 8]
    f2, ax2 = plt.subplots(len(orig_channels), 1, figsize=(8, 3.2 * len(orig_channels)))
    for k, ch in enumerate(orig_channels):
        ax = ax2[k]
        for e in ORDER:
            if e not in stats:
                continue
            m, ci = stats[e]
            ax.plot(x, m[:, ch], '-o', ms=3, color=COLORS.get(e, 'gray'), label=DISP.get(e, e))
            ax.fill_between(x, m[:, ch] - ci[:, ch], m[:, ch] + ci[:, ch], color=COLORS.get(e, 'gray'), alpha=0.18)
        ax.plot(x, bm[:, ch], 'k--', lw=1.2, label='no entity')
        ax.set_title(f'Channel {ch}')
        ax.set_xlabel('Timestep'); ax.set_ylabel('Activation')
        ax.legend(loc='lower right', fontsize=8)
    f2.suptitle(f'Conv4a parallel-rollout trajectories, mean ±95% CI ({a.num_seeds} seeds)', y=1.0)
    f2.tight_layout()
    out2 = os.path.join(a.output_dir, 'fig41_original_style.pdf')
    f2.savefig(out2, bbox_inches='tight'); f2.savefig(out2.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')

    json.dump({'top_channels_by_variance': order[:8].tolist(), 'orig_channels': orig_channels,
               'T': int(T), 'num_seeds': a.num_seeds, 'inter_entity_variance': inter_var.tolist()},
              open(os.path.join(a.output_dir, 'fig41_meta.json'), 'w'), indent=2)
    print('saved', out, 'and', out2)


if __name__ == '__main__':
    main()
