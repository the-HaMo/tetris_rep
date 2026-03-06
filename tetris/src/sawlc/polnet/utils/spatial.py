"""
Spatial utilities for grid-based protein placement and Tetris algorithm.
"""

import numpy as np
import random


def generate_grid_positions(voi_shape, protein_size, spacing_multiplier=1.2):
    """
    Generate regular grid positions within the VOI.
    
    Args:
        voi_shape (tuple): Shape of VOI (nx, ny, nz) in voxels
        protein_size (float): Approximate size of protein in voxels
        spacing_multiplier (float): Multiplier for spacing (1.2 = 20% spacing)
    
    Returns:
        list: Array of (x, y, z) grid positions
    """
    
    if protein_size <= 0:
        raise ValueError("protein_size must be positive")
    if spacing_multiplier <= 1.0:
        raise ValueError("spacing_multiplier must be > 1.0")
    
    spacing = protein_size * spacing_multiplier
    
    nx, ny, nz = voi_shape
    
    grid_positions = []
    
    # Generate grid
    x_positions = np.arange(protein_size / 2, nx, spacing)
    y_positions = np.arange(protein_size / 2, ny, spacing)
    z_positions = np.arange(protein_size / 2, nz, spacing)
    
    for x in x_positions:
        for y in y_positions:
            for z in z_positions:
                grid_positions.append((x, y, z))
    
    return grid_positions


def add_random_offset(position, offset_magnitude, protein_size):
    """
    Add random offset to position for heterogeneity.
    
    Args:
        position (tuple): (x, y, z) position
        offset_magnitude (float): Max offset magnitude (0.0-1.0 as fraction of protein_size)
        protein_size (float): Size of protein
    
    Returns:
        tuple: Noisy position (x, y, z)
    """
    
    if not (0.0 <= offset_magnitude <= 1.0):
        raise ValueError("offset_magnitude must be in [0.0, 1.0]")
    
    if protein_size <= 0:
        raise ValueError("protein_size must be positive")
    
    max_offset = offset_magnitude * protein_size
    
    offset_x = random.uniform(-max_offset, max_offset)
    offset_y = random.uniform(-max_offset, max_offset)
    offset_z = random.uniform(-max_offset, max_offset)
    
    return (
        position[0] + offset_x,
        position[1] + offset_y,
        position[2] + offset_z,
    )


def is_position_in_voi(position, voi_shape, protein_radius):
    """
    Check if position (with protein bounding box) is within VOI.
    
    Args:
        position (tuple): (x, y, z) center position
        voi_shape (tuple): (nx, ny, nz) VOI dimensions
        protein_radius (float): Radius of protein bounding sphere
    
    Returns:
        bool: True if position is valid
    """
    
    x, y, z = position
    nx, ny, nz = voi_shape
    
    # Check if center and radius are within bounds
    if x - protein_radius < 0 or x + protein_radius > nx:
        return False
    if y - protein_radius < 0 or y + protein_radius > ny:
        return False
    if z - protein_radius < 0 or z + protein_radius > nz:
        return False
    
    return True

# use in testing
def compute_grid_statistics(grid_positions):
    """
    Compute statistics about the grid.
    
    Args:
        grid_positions (list): List of (x, y, z) positions
    
    Returns:
        dict: Statistics (count, density, spacing, etc.)
    """
    
    if not grid_positions:
        return {}
    
    grid_array = np.array(grid_positions)
    
    return {
        "count": len(grid_positions),
        "mean_x": np.mean(grid_array[:, 0]),
        "mean_y": np.mean(grid_array[:, 1]),
        "mean_z": np.mean(grid_array[:, 2]),
        "std_x": np.std(grid_array[:, 0]),
        "std_y": np.std(grid_array[:, 1]),
        "std_z": np.std(grid_array[:, 2]),
    }

# use in testing 
def filter_positions_in_voi(positions, voi_shape, protein_radius):
    """
    Filter positions to keep only those within VOI.
    
    Args:
        positions (list): List of (x, y, z) positions
        voi_shape (tuple): (nx, ny, nz) VOI dimensions
        protein_radius (float): Radius of protein
    
    Returns:
        list: Filtered positions
    """
    
    valid_positions = []
    for pos in positions:
        if is_position_in_voi(pos, voi_shape, protein_radius):
            valid_positions.append(pos)
    
    return valid_positions
