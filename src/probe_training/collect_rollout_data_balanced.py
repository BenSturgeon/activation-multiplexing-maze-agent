#!/usr/bin/env python3
"""
Efficient balanced data collection from rollouts.
Stops collecting each entity type once target samples are reached.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import pickle
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import argparse

from utils import heist
from utils.helpers import load_interpretable_model, ModelActivations, observation_to_rgb, get_device, generate_action
from utils.create_intervention_mazes import create_empty_corners_maze, create_cross_maze
from utils.entity_collection_detector import detect_collections, get_collection_status


def get_entity_positions(state):
    """Get current positions of all entities in the maze."""
    entities = {}
    state_vals = state.state_vals

    for ent in state_vals["ents"]:
        entity_type = ent["type"].val
        x = ent["x"].val
        y = ent["y"].val

        # Skip invalid positions
        if x < 0 or y < 0:
            continue

        # Player (type 0)
        if entity_type == 0:
            entities['player'] = (x, y)
        # Keys (type 2)
        elif entity_type == 2:
            color = ent["image_theme"].val
            color_names = {0: 'blue_key', 1: 'green_key', 2: 'red_key'}
            if color in color_names:
                entities[color_names[color]] = (x, y)
        # Locks (type 1)
        elif entity_type == 1:
            color = ent["image_theme"].val
            color_names = {0: 'blue_lock', 1: 'green_lock', 2: 'red_lock'}
            if color in color_names:
                entities[color_names[color]] = (x, y)
        # Gems (type 9)
        elif entity_type == 9:
            entities['gem'] = (x, y)

    return entities


def determine_next_target(collection_status, entity_positions):
    """
    Determine what the next target should be based on game logic.

    Priority order:
    1. Blue key (if not collected)
    2. Blue lock (if have blue key and lock not opened)
    3. Green key (if not collected and blue lock opened)
    4. Green lock (if have green key and lock not opened)
    5. Red key (if not collected and green lock opened)
    6. Red lock (if have red key and lock not opened)
    7. Gem (after all locks opened)
    """
    # Check what's been collected/opened
    blue_key_collected = collection_status.get('blue_key', False)
    green_key_collected = collection_status.get('green_key', False)
    red_key_collected = collection_status.get('red_key', False)
    blue_lock_opened = collection_status.get('blue_lock', False)
    green_lock_opened = collection_status.get('green_lock', False)
    red_lock_opened = collection_status.get('red_lock', False)

    # Determine next target based on progression
    if not blue_key_collected and 'blue_key' in entity_positions:
        return 'blue_key'
    elif blue_key_collected and not blue_lock_opened and 'blue_lock' in entity_positions:
        return 'blue_lock'
    elif blue_lock_opened and not green_key_collected and 'green_key' in entity_positions:
        return 'green_key'
    elif green_key_collected and not green_lock_opened and 'green_lock' in entity_positions:
        return 'green_lock'
    elif green_lock_opened and not red_key_collected and 'red_key' in entity_positions:
        return 'red_key'
    elif red_key_collected and not red_lock_opened and 'red_lock' in entity_positions:
        return 'red_lock'
    elif 'gem' in entity_positions and not collection_status.get('gem', False):
        return 'gem'

    return None


def collect_balanced_dataset_efficient(model, samples_per_entity=500,
                                      collect_every=2, env_type='empty',
                                      max_steps_per_rollout=100, logger=None,
                                      max_rollouts=50000):
    """
    Efficiently collect a balanced dataset by tracking collection progress
    and stopping collection for each entity once quota is met.

    Args:
        model: Trained model for generating actions
        samples_per_entity: Target number of samples per entity type
        collect_every: Collect data every N steps
        env_type: 'empty', 'standard', or 'cross' maze environment
        max_steps_per_rollout: Maximum steps per rollout
        logger: Logger instance
        max_rollouts: Maximum number of rollouts before giving up

    Returns:
        Balanced dataset with equal samples per entity
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Label mapping - different for empty vs standard vs cross environments
    if env_type == 'empty':
        # Empty mazes only have keys and gems, no locks
        label_map = {
            'blue_key': 0,
            'green_key': 1,
            'red_key': 2,
            'gem': 3
        }
    elif env_type == 'cross':
        # Cross mazes only have keys and gems, no locks
        label_map = {
            'blue_key': 0,
            'green_key': 1,
            'red_key': 2,
            'gem': 3
        }
    else:
        # Standard mazes have both keys and locks
        label_map = {
            'blue_key': 0,
            'green_key': 1,
            'red_key': 2,
            'blue_lock': 3,
            'green_lock': 4,
            'red_lock': 5,
            'gem': 6
        }
    
    # Initialize collection tracking
    collected = {entity: {
        'observations': [],
        'activations': {layer: [] for layer in ['conv1a', 'conv2a', 'conv2b', 'conv3a', 'conv4a', 'fc1', 'fc2', 'fc3']},
        'labels': [],
        'metadata': [],
        'count': 0
    } for entity in label_map.keys()}
    
    # Track which entities have met their target
    targets_met = set()
    all_entities = set(label_map.keys())
    
    rollout_count = 0
    total_steps = 0
    
    logger.info(f"Collecting {samples_per_entity} samples per entity ({len(label_map)} entities)")
    logger.info(f"Environment type: {env_type}")
    logger.info(f"Collecting every {collect_every} steps")
    
    # Progress bar for total samples needed
    total_samples_needed = samples_per_entity * len(label_map)
    pbar = tqdm(total=total_samples_needed, desc="Collecting balanced samples")

    # Track progress for each entity
    last_progress_report = 0

    # Load viable seeds
    import os
    viable_seeds_path = os.path.join(os.path.dirname(__file__), '../utils/viable_seeds.txt')
    if os.path.exists(viable_seeds_path):
        with open(viable_seeds_path, 'r') as f:
            viable_seeds = [int(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(viable_seeds)} viable seeds")
    else:
        logger.warning("Viable seeds file not found, using random seeds")
        viable_seeds = list(range(100000))

    while targets_met != all_entities and rollout_count < max_rollouts:
        rollout_count += 1
        # Use viable seed
        seed = random.choice(viable_seeds)
        
        # Create environment
        if env_type == 'empty':
            _, venv = create_empty_corners_maze(randomize_entities=True)
            obs = venv.reset()
        elif env_type == 'cross':
            _, venv = create_cross_maze(include_locks=False)
            obs = venv.reset()
        else:  # standard
            venv = heist.create_venv(
                num=1,
                start_level=seed,
                num_levels=1,  # Use 1 to match viable seed generation
                distribution_mode="easy"
            )
            obs = venv.reset()
        
        state = heist.state_from_venv(venv, 0)

        # Initialize entity tracking
        entity_counts = None  # Will be set on first detection

        # Setup model hooks for activation extraction (once per rollout)
        model_activations = ModelActivations(model)
        layer_names = list(collected[next(iter(collected))]['activations'].keys())

        for step in range(max_steps_per_rollout):
            total_steps += 1

            # Track entity collections/openings
            entity_counts, collected_this_step = detect_collections(state, entity_counts)
            collection_status = get_collection_status(entity_counts)

            # Get current entity positions
            current_entities = get_entity_positions(state)

            # Determine next target based on collection status
            next_target = determine_next_target(collection_status, current_entities)

            # For locks, we want to collect samples when the agent HAS the key and the lock exists
            # (i.e., when the agent could potentially open it)
            if step % collect_every == 0:
                entity_to_collect = None

                # Only check for lock samples in standard environment
                if env_type == 'standard':
                    # Check if we should collect lock samples
                    if collection_status.get('blue_key', False) and not collection_status.get('blue_lock', False):
                        if 'blue_lock' in current_entities and collected['blue_lock']['count'] < samples_per_entity:
                            entity_to_collect = 'blue_lock'
                    elif collection_status.get('green_key', False) and not collection_status.get('green_lock', False):
                        if 'green_lock' in current_entities and collected['green_lock']['count'] < samples_per_entity:
                            entity_to_collect = 'green_lock'
                    elif collection_status.get('red_key', False) and not collection_status.get('red_lock', False):
                        if 'red_lock' in current_entities and collected['red_lock']['count'] < samples_per_entity:
                            entity_to_collect = 'red_lock'

                # Collect for the next target (keys/gem) if no lock to collect
                if entity_to_collect is None and next_target is not None and next_target not in ['blue_lock', 'green_lock', 'red_lock']:
                    if next_target in collected and collected[next_target]['count'] < samples_per_entity:
                        entity_to_collect = next_target

                if entity_to_collect is not None:
                    # Get activations using model forward pass
                    obs_rgb = observation_to_rgb(obs)
                    _, activations = model_activations.run_with_cache(obs_rgb, layer_names)

                    # Store observation
                    collected[entity_to_collect]['observations'].append(obs[0])
                    collected[entity_to_collect]['labels'].append(label_map[entity_to_collect])

                    # Store activations for this sample
                    for layer_name in layer_names:
                        layer_key = layer_name.replace('.', '_')
                        if layer_key in activations:
                            feat = activations[layer_key]
                            if isinstance(feat, tuple):
                                feat = feat[0]
                            collected[entity_to_collect]['activations'][layer_name].append(
                                feat.detach().cpu().numpy()
                            )

                    # Store metadata
                    collected[entity_to_collect]['metadata'].append({
                        'rollout': rollout_count,
                        'step': step,
                        'collection_status': collection_status.copy(),
                        'current_entities': list(current_entities.keys()),
                        'player_pos': current_entities.get('player', (-1, -1)),
                        'seed': seed
                    })

                    collected[entity_to_collect]['count'] += 1
                    pbar.update(1)

                    # Check if we've met target for this entity
                    if collected[entity_to_collect]['count'] >= samples_per_entity:
                        targets_met.add(entity_to_collect)
                        logger.info(f"  ✓ Completed collection for {entity_to_collect} "
                                  f"({collected[entity_to_collect]['count']} samples)")
            
            # Get action from model
            obs_rgb = observation_to_rgb(obs)
            action = generate_action(model, obs_rgb)
            
            # Take action
            obs, reward, done, info = venv.step(action)
            
            # Update state
            state = heist.state_from_venv(venv, 0)

            # Check if gem was collected (ends episode)
            if 'gem' in collected_this_step:
                break
            
            if done[0]:
                break
        
        venv.close()
        model_activations.clear_hooks()

        # Log progress regularly
        current_total = sum(v['count'] for v in collected.values())
        if current_total - last_progress_report >= 20 or rollout_count % 5 == 0:
            last_progress_report = current_total
            counts_str = ", ".join([f"{k}={v['count']}" for k, v in collected.items()])
            logger.info(f"  Rollout {rollout_count}: {counts_str}")

            # Check if we're stuck (gems taking too long)
            if rollout_count > 100 and collected['gem']['count'] < samples_per_entity * 0.1:
                logger.warning(f"  Gem collection is slow ({collected['gem']['count']}/{samples_per_entity}). "
                             f"Consider using 'empty' environment for faster collection.")
    
    pbar.close()

    # Check if we hit the rollout limit
    if rollout_count >= max_rollouts:
        logger.warning(f"\nReached maximum rollout limit ({max_rollouts})")
        logger.warning("Collection incomplete:")
        for entity, data in collected.items():
            if data['count'] < samples_per_entity:
                logger.warning(f"  {entity}: {data['count']}/{samples_per_entity} samples")

    # Convert to final format
    logger.info(f"\nCollection complete after {rollout_count} rollouts, {total_steps} total steps")
    logger.info("Final counts:")
    for entity, data in collected.items():
        status = "✓" if data['count'] >= samples_per_entity else "✗"
        logger.info(f"  {status} {entity}: {data['count']} samples (target: {samples_per_entity})")
    
    # Merge all data
    all_observations = []
    all_activations = {layer: [] for layer in collected[next(iter(collected))]['activations'].keys()}
    all_labels = []
    all_metadata = []
    
    for entity_data in collected.values():
        all_observations.extend(entity_data['observations'])
        all_labels.extend(entity_data['labels'])
        all_metadata.extend(entity_data['metadata'])
        
        for layer in all_activations.keys():
            all_activations[layer].extend(entity_data['activations'][layer])
    
    # Convert to numpy arrays
    all_observations = np.array(all_observations)
    all_labels = np.array(all_labels)
    
    for layer in all_activations.keys():
        if all_activations[layer]:
            all_activations[layer] = np.array(all_activations[layer])
    
    return all_observations, all_activations, all_labels, label_map, all_metadata


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Collect balanced rollout data')
    parser.add_argument('--samples_per_entity', type=int, default=500,
                       help='Number of samples to collect per entity type')
    parser.add_argument('--collect_every', type=int, default=2,
                       help='Collect data every N steps')
    parser.add_argument('--env_type', choices=['empty', 'standard', 'cross'], default='empty',
                       help='Environment type')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Efficient Balanced Data Collection")
    logger.info("="*60)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load model
    logger.info("Loading model...")
    if args.model_path and os.path.exists(args.model_path):
        model = load_interpretable_model(model_path=args.model_path)
    else:
        model_path = "../../base_models/full_run/model_35001.0.pt"
        if os.path.exists(model_path):
            model = load_interpretable_model(model_path=model_path)
        else:
            model = load_interpretable_model()
    
    model.eval()
    
    # Collect balanced dataset
    observations, activations, labels, label_map, metadata = collect_balanced_dataset_efficient(
        model,
        samples_per_entity=args.samples_per_entity,
        collect_every=args.collect_every,
        env_type=args.env_type,
        logger=logger
    )
    
    # Dataset saving disabled to save disk space
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # dataset_path = os.path.join(
    #     os.path.dirname(__file__),
    #     f'balanced_dataset_{args.env_type}_{timestamp}.pkl'
    # )
    #
    # with open(dataset_path, 'wb') as f:
    #     pickle.dump({
    #         'observations': observations,
    #         'activations': activations,
    #         'labels': labels,
    #         'label_map': label_map,
    #         'metadata': metadata,
    #         'samples_per_entity': args.samples_per_entity,
    #         'collect_every': args.collect_every,
    #         'env_type': args.env_type
    #     }, f)
    #
    # logger.info(f"\nDataset saved to: {dataset_path}")
    logger.info(f"Total samples: {len(observations)}")
    logger.info("Data collection complete! (Dataset not saved to disk)")


if __name__ == "__main__":
    main()