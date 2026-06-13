#!/usr/bin/env python3
"""Find 3-key seeds where the model actually completes the maze."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from src.utils.helpers import load_interpretable_model, generate_action, get_device
from src.utils import heist
from src.utils.entity_collection_detector import get_entity_counts, detect_collections

THREE_KEY_SEEDS = [2, 11, 17, 18, 31, 34, 46, 55, 58, 64, 65, 66, 67, 89, 95, 102, 107, 114, 121, 124]

model = load_interpretable_model(model_path="base_models/full_run/model_35001.0.pt")
model.eval()

for seed in THREE_KEY_SEEDS:
    venv = heist.create_venv(num=1, start_level=seed, num_levels=1)
    obs = venv.reset()

    state = heist.state_from_venv(venv, 0)
    prev_counts = get_entity_counts(state)
    gem_exists = prev_counts.get(3, 0) > 0
    collections = []

    for step in range(400):
        action = generate_action(model, obs)
        obs, reward, done, info = venv.step(action)

        done_flag = done[0] if isinstance(done, np.ndarray) else done
        state = heist.state_from_venv(venv, 0)
        current_counts, collected = detect_collections(
            state, prev_counts, episode_done=bool(done_flag), initial_gem_exists=gem_exists
        )
        prev_counts = current_counts
        collections.extend(collected)

        if done_flag:
            break

    print(f"Seed {seed}: {len(collections)} collections: {collections}, steps={step}")
