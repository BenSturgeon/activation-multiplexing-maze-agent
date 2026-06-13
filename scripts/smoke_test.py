#!/usr/bin/env python3
"""Quick smoke test: load model, create env, run a few rollouts with intervention."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import random

print("=== Smoke Test ===")
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from utils.helpers import load_interpretable_model_fast, get_device, generate_action, ModelActivations
from utils.create_intervention_mazes import create_expanded_cross_maze
from utils.entity_collection_detector import get_entity_counts, detect_collections
from utils import heist

device = get_device()
print(f"Device: {device}")

# Load model
model = load_interpretable_model_fast("base_models/full_run/model_35001.0.pt").to(device)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Run 3 rollouts: baseline, positive offset, negative offset
test_seed = 42
for label, offset in [("baseline", 0.0), ("positive +3.7", 3.7), ("negative -3.0", -3.0)]:
    random.seed(test_seed)
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)

    obs_list, venv = create_expanded_cross_maze(seed=test_seed)
    obs = obs_list[0]
    state = heist.state_from_venv(venv, 0)
    initial_counts = get_entity_counts(state)

    hook = None
    if offset != 0.0:
        def make_hook(off):
            def modify(module, input, output):
                output[:, :] += off
                return output
            return modify
        hook = model.conv4a.register_forward_hook(make_hook(offset))

    first_collected = None
    for step in range(200):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            action = generate_action(model, obs_tensor)
        obs, _, done, info = venv.step(np.array([action]))

        state = heist.state_from_venv(venv, 0)
        current_counts = get_entity_counts(state)
        collections = detect_collections(initial_counts, current_counts)
        if collections and first_collected is None:
            first_collected = collections[0]

        if done[0]:
            break

    if hook:
        hook.remove()
    venv.close()

    print(f"  {label:20s} -> first collected: {first_collected or 'nothing'} (steps: {step+1})")

print("\n=== Smoke Test PASSED ===")
