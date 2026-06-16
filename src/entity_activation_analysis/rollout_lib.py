#!/usr/bin/env python3
"""
Shared, behavior-preserving primitives for conv4a rollout/capture experiments.

Single home for the loop that no_entity_trajectory_control.py and
channel_entity_tuning.py used to each reimplement. Keep these pure and stable:
the verify harness (verify_rollout.py) pins their numerical behavior.
"""

import os
import numpy as np
import torch

from src.utils.helpers import (
    load_interpretable_model, ModelActivations, get_device,
    generate_action, observation_to_rgb,
)
from src.utils import heist

CONV4A = 'conv4a'
DEFAULT_CHECKPOINT = 'model_35001.0.pt'


def load_model(checkpoint=DEFAULT_CHECKPOINT):
    """Load the interpretable model + an ModelActivations wrapper, on device."""
    mp = os.path.join(os.path.dirname(__file__),
                      f'../../base_models/full_run/{checkpoint}')
    model = load_interpretable_model(model_path=mp) if os.path.exists(mp) \
        else load_interpretable_model()
    model.eval().to(get_device())
    return model, ModelActivations(model)


def strip_entities(venv):
    """Remove all target entities (gem + keys + locks) from a live venv in place.
    Returns the post-reset observation."""
    state = heist.state_from_venv(venv, 0)
    state.remove_gem()
    state.delete_keys()
    state.delete_locks()
    sb = state.state_bytes
    if sb is None:
        raise RuntimeError("state_bytes is None after stripping entities")
    venv.env.callmethod("set_state", [sb])
    return venv.reset()


def capture_pooled(model_activations, obs, layer=CONV4A, device=None):
    """conv4a per-channel spatial-mean (the pooled activation vector) for one obs.
    Returns ndarray (C,) or None if the layer is absent from the cache."""
    device = device or get_device()
    obs_rgb = observation_to_rgb(obs if obs.ndim == 3 else obs[0])
    t = torch.tensor(obs_rgb, dtype=torch.float32).unsqueeze(0).to(device)
    _, acts = model_activations.run_with_cache(t, [layer])
    if layer not in acts:
        return None
    return torch.mean(acts[layer], dim=(1, 2)).cpu().numpy().ravel()


def rollout(model, model_activations, venv, obs, max_steps=120,
            layer=CONV4A, on_capture=None):
    """Run a rollout, capturing the pooled activation at every step.

    on_capture(step, pooled, state_before_action) is called for each captured
    step (state is read-only; use it for collection/phase tracking). Returns
    pooled_per_step ndarray of shape (T, C).

    Order matches the original scripts exactly: capture the current obs, then
    act/step. Read-only state_from_venv does not perturb RNG or activations, so
    numerical behavior is preserved (pinned by verify_rollout.py).
    """
    device = get_device()
    obs = obs[0] if obs.ndim == 4 else obs
    pooled_per_step = []
    for step in range(max_steps):
        pooled = capture_pooled(model_activations, obs, layer, device)
        if pooled is not None:
            pooled_per_step.append(pooled)
            if on_capture is not None:
                on_capture(step, pooled, heist.state_from_venv(venv, 0))
        action = generate_action(model, obs)
        obs, _, done, _ = venv.step(action)
        obs = obs[0]
        if np.array(done).ravel()[0]:
            break
    return np.array(pooled_per_step)


def make_venv(seed, num_levels=1, strip=False):
    """Standard heist venv at a fixed seed; optionally strip all entities.
    Returns (venv, obs)."""
    venv = heist.create_venv(num=1, start_level=seed, num_levels=num_levels)
    obs = strip_entities(venv) if strip else venv.reset()
    return venv, obs
