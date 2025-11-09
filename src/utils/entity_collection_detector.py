#!/usr/bin/env python3
"""
Entity collection detection utilities for Heist environment.
Provides functions to track when entities are collected during gameplay.
"""

from typing import Dict, List, Tuple, Optional
from utils.heist import EnvState

# Entity codes and their meanings
ENTITY_CODES = {
    3: "gem",
    4: "blue_key",
    5: "green_key",
    6: "red_key",
    7: "blue_lock",
    8: "green_lock",
    9: "red_lock"
}

# Entity type and theme mappings
ENTITY_TYPE_THEME = {
    3: (9, None),  # gem: type 9, no theme
    4: (2, 0),     # blue_key: type 2, theme 0
    5: (2, 1),     # green_key: type 2, theme 1
    6: (2, 2),     # red_key: type 2, theme 2
    7: (1, 0),     # blue_lock: type 1, theme 0
    8: (1, 1),     # green_lock: type 1, theme 1
    9: (1, 2),     # red_lock: type 1, theme 2
}


def get_entity_counts(state: EnvState) -> Dict[int, int]:
    """
    Get the current count of each entity type from the environment state.
    
    Args:
        state: Current EnvState object
        
    Returns:
        Dict mapping entity codes to their counts
        - Keys (codes 4,5,6): count of 2 means on board + in HUD, 1 means only in HUD
        - Gem (code 3): count of 1 means on board, 0 means collected
    """
    counts = {}
    
    # Count each entity type
    for code, (entity_type, theme) in ENTITY_TYPE_THEME.items():
        if theme is None:
            # Gem has no theme parameter
            counts[code] = state.count_entities(entity_type)
        else:
            # Keys have themes
            counts[code] = state.count_entities(entity_type, theme)
    
    return counts


def detect_collections(
    current_state: EnvState,
    previous_counts: Optional[Dict[int, int]] = None,
    episode_done: bool = False,
    initial_gem_exists: bool = False
) -> Tuple[Dict[int, int], List[str]]:
    """
    Detect which entities were collected by comparing current state to previous counts.

    This is the main single-responsibility function for entity collection detection.

    NOTE: Gem collection cannot be detected via entity counts because the episode
    ends (done=True) immediately when gem is collected. Use episode_done=True to
    detect gem collection.

    Args:
        current_state: Current EnvState object
        previous_counts: Dict of entity codes to counts from previous step
                        If None, no collections will be detected (first step)
        episode_done: If True and initial_gem_exists, gem collection is detected
        initial_gem_exists: Whether gem existed at episode start (for done-based detection)

    Returns:
        Tuple of:
        - Updated counts dict for next comparison
        - List of entity names that were collected this step

    Example:
        >>> # First step - initialize
        >>> gem_exists = state.entity_exists(ENTITY_TYPES['gem'], None)
        >>> counts, collected = detect_collections(initial_state, initial_gem_exists=gem_exists)
        >>> print(counts)  # {3: 1, 4: 2, 5: 2, 6: 2}
        >>> print(collected)  # []

        >>> # Later step after picking up blue key
        >>> counts, collected = detect_collections(new_state, counts)
        >>> print(counts)  # {3: 1, 4: 1, 5: 2, 6: 2}
        >>> print(collected)  # ['blue_key']

        >>> # Episode ends (gem collected)
        >>> counts, collected = detect_collections(new_state, counts,
        ...                                         episode_done=True,
        ...                                         initial_gem_exists=True)
        >>> print(collected)  # ['gem']
    """
    # Get current counts
    current_counts = get_entity_counts(current_state)

    # If no previous counts provided, this is initialization
    if previous_counts is None:
        return current_counts, []

    # Detect collections by comparing counts
    collected_entities = []

    # Check for gem collection via done flag (can't use entity counts)
    if episode_done and initial_gem_exists:
        collected_entities.append('gem')

    for code, entity_name in ENTITY_CODES.items():
        prev_count = previous_counts.get(code, 0)
        curr_count = current_counts.get(code, 0)

        # Check if entity was collected/opened based on count change
        if code == 3:  # Gem
            # Gem collection is handled via episode_done flag above
            # (entity count method doesn't work due to immediate reset)
            pass
        elif code in [4, 5, 6]:  # Keys
            # Key is collected when count goes from 2 to 1
            # (board copy picked up, HUD copy remains)
            if prev_count == 2 and curr_count == 1:
                collected_entities.append(entity_name)
        elif code in [7, 8, 9]:  # Locks
            # Lock is opened when count goes from 1 to 0
            if prev_count == 1 and curr_count == 0:
                collected_entities.append(entity_name)

    return current_counts, collected_entities


def get_collection_status(counts: Dict[int, int]) -> Dict[str, bool]:
    """
    Get the collection status of each entity based on current counts.
    
    Args:
        counts: Dict of entity codes to their current counts
        
    Returns:
        Dict mapping entity names to whether they've been collected
    """
    status = {}
    
    for code, entity_name in ENTITY_CODES.items():
        count = counts.get(code, 0)

        if code == 3:  # Gem
            # Gem is collected if count is 0
            status[entity_name] = (count == 0)
        elif code in [4, 5, 6]:  # Keys
            # Key is collected if count is 1 (only HUD copy remains)
            status[entity_name] = (count == 1)
        elif code in [7, 8, 9]:  # Locks
            # Lock is opened if count is 0
            status[entity_name] = (count == 0)
    
    return status


def track_collection_order(
    current_state: EnvState,
    previous_counts: Optional[Dict[int, int]],
    collection_order: List[Tuple[int, str]],
    current_step: int
) -> Tuple[Dict[int, int], List[Tuple[int, str]]]:
    """
    Track entity collection order with step numbers.
    
    Args:
        current_state: Current EnvState object
        previous_counts: Previous entity counts (None for first step)
        collection_order: Existing list of (step, entity_name) tuples
        current_step: Current step number
        
    Returns:
        Tuple of:
        - Updated counts dict
        - Updated collection_order list with any new collections
        
    Example:
        >>> order = []
        >>> counts = None
        >>> 
        >>> # Step 5: collect blue key
        >>> counts, order = track_collection_order(state, counts, order, 5)
        >>> print(order)  # [(5, 'blue_key')]
        >>> 
        >>> # Step 10: collect gem
        >>> counts, order = track_collection_order(state, counts, order, 10)
        >>> print(order)  # [(5, 'blue_key'), (10, 'gem')]
    """
    # Detect collections
    current_counts, collected_this_step = detect_collections(current_state, previous_counts)
    
    # Add to collection order
    updated_order = collection_order.copy()
    for entity_name in collected_this_step:
        updated_order.append((current_step, entity_name))
    
    return current_counts, updated_order


# Utility functions for specific checks

def all_entities_collected(counts: Dict[int, int]) -> bool:
    """
    Check if all entities have been collected.
    
    Args:
        counts: Current entity counts
        
    Returns:
        True if all entities collected, False otherwise
    """
    for code in ENTITY_CODES:
        count = counts.get(code, 0)
        if code == 3:  # Gem
            if count > 0:  # Gem still on board
                return False
        else:  # Keys
            if count > 1:  # Key still on board (not just in HUD)
                return False
    return True


def count_remaining_entities(counts: Dict[int, int]) -> int:
    """
    Count how many entities are still on the board (not collected).
    
    Args:
        counts: Current entity counts
        
    Returns:
        Number of entities still on the board
    """
    remaining = 0
    for code in ENTITY_CODES:
        count = counts.get(code, 0)
        if code == 3:  # Gem
            if count > 0:
                remaining += 1
        else:  # Keys
            if count > 1:  # More than just HUD copy
                remaining += 1
    return remaining


def get_initial_entities(state: EnvState) -> List[str]:
    """
    Get list of entities present at the start of an episode.
    
    Args:
        state: Initial EnvState object
        
    Returns:
        List of entity names that are present on the board
    """
    counts = get_entity_counts(state)
    present = []
    
    for code, entity_name in ENTITY_CODES.items():
        count = counts.get(code, 0)
        if code == 3:  # Gem
            if count > 0:
                present.append(entity_name)
        else:  # Keys
            if count > 1:  # More than just HUD copy
                present.append(entity_name)
    
    return present


# For backward compatibility with existing code
def check_entity_collected(
    initial_counts: Dict[int, int],
    current_counts: Dict[int, int],
    entity_code: int
) -> bool:
    """
    Check if a specific entity has been collected.
    
    Args:
        initial_counts: Initial entity counts
        current_counts: Current entity counts  
        entity_code: Code of entity to check (3=gem, 4=blue_key, etc.)
        
    Returns:
        True if entity was collected, False otherwise
    """
    if entity_code not in ENTITY_CODES:
        return False
        
    init_count = initial_counts.get(entity_code, 0)
    curr_count = current_counts.get(entity_code, 0)
    
    if entity_code == 3:  # Gem
        # Initially present and now gone
        return init_count == 1 and curr_count == 0
    else:  # Keys
        # Initially on board (count=2) and now only in HUD (count=1)
        return init_count == 2 and curr_count == 1