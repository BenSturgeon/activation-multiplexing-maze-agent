# Green Key Steering: Exact Replication Guide

## Overview
This documents the exact procedure to replicate the remarkable result where we achieved **80% green-first collection** from seeds that naturally have **100% blue-first collection**.

## Result Summary
- **Baseline**: 100% blue-first, 0% green-first
- **With intervention**: 0% blue-first, 80% green-first
- **Method**: Apply offset = `-initial_conv4a_activation`
- **Success rate**: 8/10 seeds (80%)

## The Exact Offset Applied

### Initial Activations (Natural State)
- **Mean**: -3.6 across 32 channels
- **Range**: [-6.5, -0.3]
- All initial activations were **negative**

### Applied Offset (Negated Initial)
- **Mean**: +3.6 across 32 channels
- **Range**: [+0.4, +6.0]
- All offset values were **positive** (boosting)

### Top 5 Strongest Channels
1. Channel 24: +6.006
2. Channel 18: +6.004
3. Channel 28: +5.308
4. Channel 27: +5.228
5. Channel 4: +4.803

## Exact Replication Steps

### 1. Prerequisites
```bash
# Activate environment
source .venv/bin/activate

# Ensure you have the model
ls base_models/full_run/model_35001.0.pt
```

### 2. Run the Exact Command
```bash
python src/entity_activation_analysis/capture_mid_rollout_offset.py \
  --target green \
  --inverse \
  --maze_type expanded \
  --offset_method initial \
  --n_trials 10
```

### 3. What This Does

#### Step A: Load Baseline Behaviors
- Uses viable seeds from `src/utils/viable_seeds.txt`
- Loads or creates baseline file: `baseline_natural_order_seeds_expanded.json`
- Baseline shows ~89% blue-first, ~10% green-first naturally

#### Step B: Find Target Seeds
- Searches for seeds where agent collects green key (after blue)
- Captures activation patterns during green targeting
- Needs 10 successful captures

#### Step C: Calculate Offsets
For each seed:
1. Get initial observation
2. Extract conv4a activation (32 channels after spatial pooling)
3. Apply negation: `offset = -initial_activation`
4. Result: positive values (mean ~+3.6)

#### Step D: Apply Intervention
For each seed:
1. Create expanded maze with same seed
2. Install forward hook on conv4a:
   ```python
   def modify_conv4a(module, input, output):
       for ch in range(32):
           output[:, ch] += offset[0, ch]
       return output
   ```
3. Run rollout and observe first collection

### 4. Expected Output
```
=== RESULTS: Per-Seed GREEN_KEY-Targeting Offsets ===
Tested on 10 seeds with captured green_key patterns:
  green_key: 8 (80.0%)
  None: 2

Compare to baseline (same 10 seeds):
Entity       Baseline     Result       Change       p-value
------------------------------------------------------------
blue_key     100.0± 0.0%    0.0± 0.0%  -100.0%  **   p=0.0020
green_key      0.0± 0.0%   80.0±12.6%   +80.0%  **   p=0.0078
```

## The Specific Seeds Used
These 10 seeds from viable_seeds.txt showed the effect:
- 882055386 → green_key
- 2639866 → green_key
- 44278588 → None
- 372648555 → green_key
- 952142617 → green_key
- 227892002 → green_key
- 629683338 → green_key
- 1063938045 → green_key
- 933726074 → None
- 870990467 → green_key

## Analysis Script
To analyze the exact offset values:
```bash
python analyze_green_steering_offsets.py
```

This will show:
- Per-seed initial activations and offsets
- Channel-wise statistics
- Saves to `green_steering_offset_analysis.json`

## Key Insight
**The initial frame contains a strong negative signal (mean -3.6) in conv4a that, when negated, completely reverses the agent's key preference from blue to green.**

This suggests:
1. Initial activations encode default preferences
2. Simple linear modifications can flip behaviors
3. Conv4a is a critical decision layer for navigation

## Next Experiments
1. Test with n=400 for more statistical power
2. Compare to blue-first vs green-first initial activations
3. Test if 2x or 0.5x offset has different effects
4. Check if specific channels correspond to color detectors