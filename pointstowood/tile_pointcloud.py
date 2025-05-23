#!/usr/bin/env python3
"""
Spatial tiling utility for large point clouds.
Splits point clouds into spatial tiles for processing.
"""

import os
import sys
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.io import load_file, save_file

def tile_point_cloud(input_file, tile_size=50.0, overlap=0.1, output_dir=None):
    """
    Split point cloud into spatial tiles.
    
    Args:
        input_file (str): Path to input point cloud
        tile_size (float): Size of each tile in meters
        overlap (float): Overlap fraction between tiles (0.0-0.5)
        output_dir (str): Output directory (default: same as input)
    
    Returns:
        list: Paths to generated tile files
    """
    print(f"Loading point cloud: {input_file}")
    pc, headers = load_file(input_file, additional_headers=True)
    
    # Get bounding box
    min_x, max_x = pc['x'].min(), pc['x'].max()
    min_y, max_y = pc['y'].min(), pc['y'].max()
    
    print(f"Point cloud bounds: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
    
    # Calculate overlap in meters
    overlap_dist = tile_size * overlap
    step_size = tile_size - overlap_dist
    
    # Generate tile boundaries
    x_tiles = np.arange(min_x, max_x + tile_size, step_size)
    y_tiles = np.arange(min_y, max_y + tile_size, step_size)
    
    print(f"Creating {len(x_tiles)} x {len(y_tiles)} = {len(x_tiles) * len(y_tiles)} tiles")
    print(f"Tile size: {tile_size}m, Overlap: {overlap*100:.1f}% ({overlap_dist:.1f}m)")
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    tile_files = []
    tiles_created = 0
    
    for i, x_start in enumerate(x_tiles[:-1]):
        for j, y_start in enumerate(y_tiles[:-1]):
            x_end = x_start + tile_size
            y_end = y_start + tile_size
            
            # Filter points in this tile
            mask = ((pc['x'] >= x_start) & (pc['x'] < x_end) & 
                   (pc['y'] >= y_start) & (pc['y'] < y_end))
            
            tile_pc = pc[mask].copy()
            
            # Skip empty tiles
            if len(tile_pc) == 0:
                continue
                
            # Skip tiles with very few points
            if len(tile_pc) < 100:
                print(f"Skipping tile {i:02d}_{j:02d} - only {len(tile_pc)} points")
                continue
            
            # Debug: Check for duplicate column names
            if len(tile_pc.columns) != len(set(tile_pc.columns)):
                print(f"Warning: Duplicate columns in tile {i:02d}_{j:02d}: {list(tile_pc.columns)}")
                # Remove duplicate columns
                tile_pc = tile_pc.loc[:, ~tile_pc.columns.duplicated()]
            
            # Save tile
            tile_filename = f"{base_name}_tile_{i:02d}_{j:02d}.ply"
            tile_path = os.path.join(output_dir, tile_filename)
            
            # Check if this is raycloud format
            is_raycloud = any(col in tile_pc.columns for col in ['time', 'nx', 'ny', 'nz', 'alpha'])
            
            # Filter headers to only include columns that exist in this tile
            tile_headers = [h for h in headers if h in tile_pc.columns]
            
            save_file(tile_path, tile_pc, additional_fields=tile_headers, 
                     preserve_raycloud_format=is_raycloud)
            
            tile_files.append(tile_path)
            tiles_created += 1
            
            print(f"Created tile {i:02d}_{j:02d}: {len(tile_pc):,} points -> {tile_filename}")
    
    print(f"\nTiling complete: {tiles_created} tiles created")
    print(f"Original points: {len(pc):,}")
    
    # Count total points in tiles for verification
    total_tile_points = 0
    for f in tile_files:
        try:
            tile_pc = load_file(f, additional_headers=False)
            total_tile_points += len(tile_pc)
        except Exception as e:
            print(f"Warning: Could not verify tile {f}: {e}")
    
    print(f"Total points in tiles: {total_tile_points:,}")
    
    return tile_files

def merge_predictions(tile_files, output_file, remove_tiles=False):
    """
    Merge prediction results from multiple tiles back into single file.
    
    Args:
        tile_files (list): List of tile file paths with predictions
        output_file (str): Output merged file path
        remove_tiles (bool): Remove individual tile files after merging
    """
    print(f"Merging {len(tile_files)} tiles into {output_file}")
    
    merged_pc = []
    total_points = 0
    
    for tile_file in tile_files:
        if os.path.exists(tile_file):
            pc, headers = load_file(tile_file, additional_headers=True)
            merged_pc.append(pc)
            total_points += len(pc)
            print(f"Added {len(pc):,} points from {os.path.basename(tile_file)}")
        else:
            print(f"Warning: Tile file not found: {tile_file}")
    
    if merged_pc:
        import pandas as pd
        final_pc = pd.concat(merged_pc, ignore_index=True)
        
        # Check if this is raycloud format
        is_raycloud = any(col in final_pc.columns for col in ['time', 'nx', 'ny', 'nz', 'alpha'])
        
        save_file(output_file, final_pc, additional_fields=headers, 
                 preserve_raycloud_format=is_raycloud)
        
        print(f"Merged {total_points:,} points into {output_file}")
        
        # Optionally remove tile files
        if remove_tiles:
            for tile_file in tile_files:
                if os.path.exists(tile_file):
                    os.remove(tile_file)
            print("Removed individual tile files")
    
    return final_pc

def main():
    parser = argparse.ArgumentParser(description="Tile large point clouds for processing")
    parser.add_argument('input', help='Input point cloud file')
    parser.add_argument('--tile-size', type=float, default=50.0, 
                       help='Size of each tile in meters (default: 50.0)')
    parser.add_argument('--overlap', type=float, default=0.1,
                       help='Overlap fraction between tiles (default: 0.1)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: same as input)')
    parser.add_argument('--merge', nargs='+', 
                       help='Merge tile files back into single file')
    parser.add_argument('--merge-output', type=str,
                       help='Output file for merged results')
    parser.add_argument('--remove-tiles', action='store_true',
                       help='Remove individual tiles after merging')
    
    args = parser.parse_args()
    
    if args.merge:
        # Merge mode
        if not args.merge_output:
            args.merge_output = args.input.replace('.ply', '_merged.ply')
        merge_predictions(args.merge, args.merge_output, args.remove_tiles)
    else:
        # Tiling mode
        tile_files = tile_point_cloud(args.input, args.tile_size, args.overlap, args.output_dir)
        
        print(f"\nGenerated tiles:")
        for tile_file in tile_files:
            print(f"  {tile_file}")
        
        print(f"\nTo process tiles, run:")
        print(f"  for tile in {os.path.dirname(args.input)}/*_tile_*.ply; do")
        print(f"    python predict.py --point-cloud \"$tile\" --preserve_raycloud_format")
        print(f"  done")

if __name__ == '__main__':
    main()