#!/usr/bin/env python3
"""Find seeds where the maze has all 3 keys."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import heist
from src.utils.entity_collection_detector import get_entity_counts

seeds_with_3_keys = []

for seed in range(500):
    venv = heist.create_venv(num=1, start_level=seed, num_levels=1)
    venv.reset()
    state = heist.state_from_venv(venv, 0)
    counts = get_entity_counts(state)

    num_keys = sum(1 for k in [4, 5, 6] if counts.get(k, 0) >= 2)
    if num_keys == 3:
        seeds_with_3_keys.append(seed)

    if seed % 100 == 0:
        print(f"Tested {seed}... found {len(seeds_with_3_keys)} so far")

print(f"\nFound {len(seeds_with_3_keys)} seeds with 3 keys:")
print(seeds_with_3_keys)

with open("/root/project/3key_seeds.txt", "w") as f:
    for s in seeds_with_3_keys:
        f.write(f"{s}\n")
