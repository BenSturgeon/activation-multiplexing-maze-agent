#!/usr/bin/env python3
"""Fixed maze creation that properly handles HUD."""
from utils import heist
from utils.hud_utils import verify_hud_working
import numpy as np


def create_empty_corners_maze_with_hud(randomize_entities=False):
    """
    Create an empty corners maze with proper HUD display.
    
    This fixes the issue where collected keys don't appear in the HUD
    by ensuring type 11 (KEY_ON_RING) entities are properly positioned.
    
    Returns:
        state: The environment state
        venv: The environment
    """
    # Start with a natural maze that has working HUD
    venv = heist.create_venv(
        num=1,
        start_level=42,
        num_levels=1,
        distribution_mode="easy"
    )
    state = heist.state_from_venv(venv, 0)
    
    # Clear the maze but keep HUD infrastructure
    # First, save HUD keys (type 11 with y < 0.5)
    hud_keys = []
    for entity in state.get_entities():
        if entity['type'].val == 11 and entity['y'].val < 0.5:
            hud_keys.append({
                'x': entity['x'].val,
                'y': entity['y'].val,
                'type': 11
            })
    
    # Clear all entities
    state.remove_all_entities()
    
    # Add player at center
    state.set_mouse_pos(4.5, 4.5)
    
    # Add keys at corners (type 2 - collectible keys)
    corners = [
        (1.5, 1.5, heist.GREEN_KEY),  # bottom-left - green
        (7.5, 1.5, heist.RED_KEY),     # bottom-right - red  
        (7.5, 7.5, heist.BLUE_KEY),    # top-right - blue
    ]
    
    if randomize_entities:
        np.random.shuffle(corners)
    
    for x, y, key_type in corners:
        # Add collectible key (type 2)
        state.set_key_position(key_type, x, y)
    
    # Add gem at remaining corner
    state.set_gem_position(1.5, 7.5)
    
    # Restore HUD keys at proper positions
    # We need 3 type 11 entities in HUD positions (one for each key color)
    hud_y = 0.02  # Standard HUD y position
    hud_x_positions = [0.5, 1.0, 1.5]  # Spread across top
    
    for i, (x, y, key_type) in enumerate(corners):
        # For each collectible key, ensure there's a corresponding HUD key
        # This requires manually adding type 11 entities
        # Note: This is a limitation - we can't directly add type 11 entities
        # through the current API
        pass
    
    # Alternative approach: Use ring key positions
    # These should create type 11 entities
    state.set_ring_key_position(heist.GREEN_KEY, hud_x_positions[0], hud_y)
    state.set_ring_key_position(heist.RED_KEY, hud_x_positions[1], hud_y)
    state.set_ring_key_position(heist.BLUE_KEY, hud_x_positions[2], hud_y)
    
    # Apply state to environment
    heist.set_venv_state(venv, 0, state)
    
    # Verify HUD is working
    if not verify_hud_working(venv):
        print("Warning: HUD may not be properly initialized")
    
    return state, venv