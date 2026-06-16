#!/usr/bin/env python3
"""
Behavior-preservation check for the rollout/capture refactor.

Compares two no_entity_control.json artifacts (golden vs new) on the per-channel
means for both conditions and the headline scalars. Pure stdlib -- no numpy or
env/procgen needed, so it runs anywhere. Exit 0 = PASS.

Usage:
    python verify_rollout.py <golden.json> <new.json> [atol]
"""

import json
import sys


def _maxdiff(a, b):
    if len(a) != len(b):
        return float("inf")
    return max(abs(x - y) for x, y in zip(a, b))


def compare(golden_path, new_path, atol=1e-5):
    g = json.load(open(golden_path))
    n = json.load(open(new_path))
    ok = True
    for cond in ("standard", "no_entity"):
        d = _maxdiff(g["summary"][cond]["per_channel_mean"],
                     n["summary"][cond]["per_channel_mean"])
        passed = d <= atol
        ok = ok and passed
        print(f"  {cond:10s}: {'PASS' if passed else 'FAIL'}  max|Δ|={d:.2e}")
    for k in ("standard_mean_activation", "no_entity_mean_activation",
              "standard_first20_mean", "no_entity_first20_mean"):
        d = abs(g["headline"][k] - n["headline"][k])
        passed = d <= atol
        ok = ok and passed
        print(f"  {k:26s}: {'PASS' if passed else 'FAIL'}  Δ={d:.2e}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    atol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    sys.exit(0 if compare(sys.argv[1], sys.argv[2], atol) else 1)
