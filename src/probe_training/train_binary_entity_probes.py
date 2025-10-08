#!/usr/bin/env python3
"""Train binary probes for each entity (entity vs background) for targeted ablations."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import time
import logging
from tqdm import tqdm
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.helpers import load_interpretable_model, ModelActivations, observation_to_rgb, get_device
from procgen import ProcgenEnv
from collect_rollout_data_balanced import collect_balanced_dataset_efficient


class ProbeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BinaryEntityProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        if input_dim < 1000:
            hidden_dim = min(hidden_dim, input_dim * 2)

        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )

    def forward(self, x):
        return self.probe(x)


def extract_all_features_single_pass(observations, model, layer_names, logger=None):
    """Extract features from ALL layers in a single pass through the data, keeping channel structure for conv layers."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Extracting features from {len(layer_names)} layers in single pass...")

    model_activations = ModelActivations(model)
    all_features = {layer: [] for layer in layer_names}

    batch_size = 32
    for i in tqdm(range(0, len(observations), batch_size), desc="Feature extraction"):
        batch = observations[i:i+batch_size]
        batch_rgb = observation_to_rgb(batch)

        _, activations = model_activations.run_with_cache(batch_rgb, layer_names)

        for layer_name in layer_names:
            layer_key = layer_name.replace('.', '_')
            if layer_key in activations:
                feat = activations[layer_key]
                if isinstance(feat, tuple):
                    feat = feat[0]

                # Keep channel structure for conv layers
                if 'conv' in layer_name and feat.ndim == 4:
                    # Keep as (batch, channels, height, width)
                    all_features[layer_name].append(feat.detach().cpu().numpy())
                else:
                    # Flatten FC layers
                    if feat.ndim > 2:
                        feat = feat.reshape(feat.shape[0], -1)
                    all_features[layer_name].append(feat.detach().cpu().numpy())

    model_activations.clear_hooks()

    for layer_name in layer_names:
        if all_features[layer_name]:
            all_features[layer_name] = np.concatenate(all_features[layer_name], axis=0)
            logger.info(f"  {layer_name}: shape {all_features[layer_name].shape}")
        else:
            logger.warning(f"  {layer_name}: no features extracted")
            all_features[layer_name] = None

    return all_features


def train_channel_binary_probe(channel_features, labels):
    """Train a simple linear binary probe for a specific channel using sklearn."""
    # Flatten spatial dimensions if present
    if channel_features.ndim > 1:
        channel_features = channel_features.reshape(channel_features.shape[0], -1)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        channel_features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train logistic regression
    clf = LogisticRegression(max_iter=100, random_state=42, class_weight='balanced')
    try:
        clf.fit(X_train, y_train)
        val_acc = clf.score(X_val, y_val) * 100
    except:
        val_acc = 50.0  # Random chance if training fails

    return val_acc, clf


def train_binary_probe(features, labels, entity_name, num_epochs=50):
    """Train a binary probe for a specific entity (entity vs background)."""
    n_samples = len(features)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)

    train_features = features[indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    val_features = features[indices[n_train:]]
    val_labels = labels[indices[n_train:]]

    train_dataset = ProbeDataset(train_features, train_labels)
    val_dataset = ProbeDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_dim = features.shape[1]
    probe = BinaryEntityProbe(input_dim)
    
    device = get_device()
    probe = probe.to(device)
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_probe_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        probe.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()
        
        probe.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = probe(batch_features)
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_probe_state = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Return both accuracy and the best model state
    return best_val_acc, best_probe_state, input_dim


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Loading model...")
    # Use checkpoint 35001
    model_path = "../base_models/full_run/model_35001.0.pt"
    logger.info(f"Using checkpoint 35001: {model_path}")
    model = load_interpretable_model(model_path=model_path)
    model.eval()

    # Collect data from rollouts at runtime
    logger.info("Collecting data from rollouts (2000 samples per entity)...")
    observations, activations, labels, label_map, metadata = collect_balanced_dataset_efficient(
        model=model,
        samples_per_entity=2000,  # Full dataset for robust per-channel training
        collect_every=2,
        env_type='standard',  # Use standard mazes to include locks
        max_steps_per_rollout=100,
        logger=logger,
        max_rollouts=100000
    )

    logger.info(f"Dataset collected: {len(observations)} samples")
    logger.info(f"Label map: {label_map}")
    
    # Train probes for multiple layers
    layer_names = ['conv2a', 'conv3a', 'conv4a', 'fc1', 'fc2', 'fc3']

    start_time = time.time()
    all_features = extract_all_features_single_pass(observations, model, layer_names, logger)
    extraction_time = time.time() - start_time
    logger.info(f"Feature extraction completed in {extraction_time:.1f} seconds")

    # Create binary labels for each entity (including locks)
    # Get all entities from label_map
    entities_of_interest = list(label_map.keys())
    binary_labels = {}

    logger.info("\nCreating binary labels for each entity...")
    for entity in entities_of_interest:
        entity_idx = label_map[entity]
        # Create binary labels: 1 if entity, 0 otherwise
        binary_labels[entity] = (labels == entity_idx).astype(np.int64)
        entity_count = binary_labels[entity].sum()
        logger.info(f"  {entity}: {entity_count} positive samples out of {len(labels)}")

    results = {}

    # Create directory for saving probes with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_dir = os.path.join(os.path.dirname(__file__), f'binary_probes_{timestamp}')
    os.makedirs(probe_dir, exist_ok=True)
    logger.info(f"Saving probes to: {probe_dir}")

    logger.info("\nTraining binary probes...")
    for layer_name in layer_names:
        if all_features[layer_name] is not None:
            features = all_features[layer_name]
            results[layer_name] = {}

            # Handle conv layers (per-channel probes)
            if 'conv' in layer_name and features.ndim == 4:
                n_channels = features.shape[1]
                logger.info(f"\n{layer_name}: Training per-channel probes ({n_channels} channels)...")

                for entity in entities_of_interest:
                    logger.info(f"  Training {entity} probes...")
                    results[layer_name][entity] = {
                        'channel_accuracies': [],
                        'n_channels': n_channels
                    }

                    for channel_idx in tqdm(range(n_channels), desc=f"  {entity} channels"):
                        # Get features for this channel
                        channel_features = features[:, channel_idx]  # Shape: (n_samples, height, width)

                        # Train binary probe for this channel
                        acc, clf = train_channel_binary_probe(channel_features, binary_labels[entity])
                        results[layer_name][entity]['channel_accuracies'].append(acc)

                    # Calculate statistics
                    accs = results[layer_name][entity]['channel_accuracies']
                    results[layer_name][entity]['mean_accuracy'] = np.mean(accs)
                    results[layer_name][entity]['std_accuracy'] = np.std(accs)
                    results[layer_name][entity]['max_accuracy'] = np.max(accs)
                    results[layer_name][entity]['best_channel'] = int(np.argmax(accs))

                    logger.info(f"    Mean={np.mean(accs):.1f}%, Max={np.max(accs):.1f}% (ch {np.argmax(accs)})")

            # Handle FC layers (train on full layer for now)
            else:
                logger.info(f"\n{layer_name}: Training full-layer probes...")

                # Flatten features if needed
                if features.ndim > 2:
                    features = features.reshape(features.shape[0], -1)

                for entity in entities_of_interest:
                    logger.info(f"  Training {entity} probe...")

                    try:
                        best_acc, probe_state, input_dim = train_binary_probe(
                            features, binary_labels[entity], entity, num_epochs=50
                        )
                        results[layer_name][entity] = {
                            'accuracy': best_acc,
                            'dim': features.shape[1]
                        }

                        # Save the probe checkpoint
                        probe_path = os.path.join(probe_dir, f'{layer_name}_{entity}_probe.pt')
                        torch.save({
                            'probe_state_dict': probe_state,
                            'input_dim': input_dim,
                            'entity': entity,
                            'accuracy': best_acc,
                            'layer_name': layer_name
                        }, probe_path)

                        logger.info(f"    Accuracy = {best_acc:.2f}%")
                    except Exception as e:
                        logger.error(f"    Error: {e}")
                        results[layer_name][entity] = {'accuracy': 0.0, 'dim': features.shape[1]}
        else:
            results[layer_name] = {'error': 'No features extracted'}
    
    results_file = os.path.join(probe_dir, 'probe_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("BINARY PROBE RESULTS SUMMARY")
    print("="*80)

    for layer_name in layer_names:
        if layer_name in results and isinstance(results[layer_name], dict) and 'error' not in results[layer_name]:
            print(f"\n{layer_name}:")
            print("-" * 50)

            if 'conv' in layer_name:
                # Conv layers - show per-channel statistics
                print(f"{'Entity':<12} {'Mean Acc':<10} {'Max Acc':<10} {'Best Ch':<10}")
                for entity in entities_of_interest:
                    if entity in results[layer_name]:
                        r = results[layer_name][entity]
                        if 'mean_accuracy' in r:
                            print(f"{entity:<12} {r['mean_accuracy']:6.1f}%     {r['max_accuracy']:6.1f}%     ch {r['best_channel']:<8}")
            else:
                # FC layers - show full layer accuracy
                print(f"{'Entity':<12} {'Accuracy':<10} {'Dim':<8}")
                for entity in entities_of_interest:
                    if entity in results[layer_name]:
                        r = results[layer_name][entity]
                        if 'accuracy' in r:
                            print(f"{entity:<12} {r['accuracy']:6.2f}%     {r['dim']:<8}")

    print("="*60)
    print(f"Total time: {time.time() - start_time:.1f}s (extraction: {extraction_time:.1f}s)")
    print(f"Results saved to: {results_file}")
    print(f"Binary probe checkpoints saved to: {probe_dir}")


if __name__ == "__main__":
    main()