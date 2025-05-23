import datetime
from tile_pointcloud import tile_point_cloud, merge_predictions
start = datetime.datetime.now()
import resource
import os
import os.path as OP
import argparse
from src.preprocessing import *
from src.predicter import SemanticSegmentation
from tqdm import tqdm
import torch
import shutil
import sys
import numpy as np
import re
from src.io import load_file, save_file

def set_num_threads(num_threads):
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

'''
Minor functions-------------------------------------------------------------------------------------------------------------
'''

def get_path(location_in_pointstowood: str = "") -> str:
    current_wdir = os.getcwd()
    match = re.search(r'PointsToWood.*?pointstowood', current_wdir, re.IGNORECASE)
    if not match:
        raise ValueError('"PointsToWood/pointstowood" not found in the current working directory path')
    last_index = match.end()
    output_path = current_wdir[:last_index]
    if location_in_pointstowood:
        output_path = os.path.join(output_path, location_in_pointstowood)
    return output_path.replace("\\", "/")

def preprocess_point_cloud_data(df):
    df.columns = df.columns.str.lower()
    columns_to_drop = ['label', 'pwood', 'pleaf']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = df.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x)
    
    # For raycloud format, use alpha channel as intensity/reflectance
    if 'alpha' in df.columns and 'reflectance' not in df.columns and 'intensity' not in df.columns:
        # Store original alpha values before using as reflectance
        df['original_alpha'] = df['alpha'].copy()
        df = df.rename(columns={'alpha': 'reflectance'})
        print('Raycloud format detected: using alpha channel as reflectance')
    else:
        df = df.rename(columns={'refl': 'reflectance', 'intensity': 'reflectance'})
    
    headers = [header for header in df.columns[3:] if header not in columns_to_drop]
    if 'reflectance' not in df.columns:
        df['reflectance'] = np.zeros(len(df))
        print('No reflectance detected, column added with zeros.')
    else:
        print('Reflectance detected')
    cols = list(df.columns)
    if 'reflectance' in cols:
        cols.insert(3, cols.pop(cols.index('reflectance')))
        df = df[cols]
    return df, headers, 'reflectance' in df.columns

'''
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
'''

def main():
    """Main entry point for the prediction script."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default=[], nargs='+', type=str, help='list of point cloud files')    
    parser.add_argument('--odir', type=str, default='.', help='output directory')
    parser.add_argument('--batch_size', default=8, type=int, help="If you get CUDA errors, try lowering this.")
    parser.add_argument('--num_procs', default=-1, type=int, help="Number of CPU cores you want to use. If you run out of RAM, lower this.")
    parser.add_argument('--resolution', type=float, default=0.01, help='Resolution to which point cloud is downsampled [m]')
    parser.add_argument('--grid_size', type=float, nargs='+', default=[2.0, 4.0], help='Grid sizes for voxelization')
    parser.add_argument('--min_pts', type=int, default=128, help='Minimum number of points in voxel')
    parser.add_argument('--max_pts', type=int, default=16384, help='Maximum number of points in voxel')
    parser.add_argument('--model', type=str, default='model.pth', help='path to candidate model')
    parser.add_argument('--is-wood', default=0.5, type=float, help='a probability above which points within KNN are classified as wood')
    parser.add_argument('--any-wood', default=1, type=float, help='a probability above which ANY point within KNN is classified as wood')
    parser.add_argument('--output_fmt', default='ply', help="file type of output")
    parser.add_argument('--preserve_raycloud_format', action='store_true', help="preserve raycloud format and only save raycloud fields")
    parser.add_argument('--tile-size', type=float, default=0.0, help='Tile size in meters for input tiling (0 = no tiling)')
    parser.add_argument('--tile-overlap', type=float, default=0.1, help='Overlap fraction between tiles (0.0-0.5)')
    parser.add_argument('--auto-merge', action='store_true', help='Automatically merge tile results into single output file')
    parser.add_argument('--verbose', action='store_true', help="print stuff")

    args = parser.parse_args()

    # Configure the number of threads based on args.num_procs
    if args.num_procs == -1:
        num_threads = os.cpu_count()
    else:
        num_threads = args.num_procs

    set_num_threads(num_threads)

    if args.verbose:
        print('\n---- parameters used ----')
        for k, v in args.__dict__.items():
            if k == 'pc': v = '{} points'.format(len(v))
            if k == 'global_shift': v = v.values
            print('{:<35}{}'.format(k, v)) 

    args.wdir = get_path()
    args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'
    args.reflectance = False

    '''
    Sanity check---------------------------------------------------------------------------------------------------------
    '''
    if args.point_cloud == '':
        raise Exception('no input specified, please specify --point-cloud')
    
    for point_cloud_file in args.point_cloud:
        if not os.path.isfile(point_cloud_file):
            raise FileNotFoundError(f'Point cloud file not found: {point_cloud_file}')
    
    '''
    If voxel file on disc, delete it.
    '''    
    
    path = OP.dirname(args.point_cloud[0])
    args.vxfile = OP.join(path, "voxels")

    if os.path.exists(args.vxfile): shutil.rmtree(args.vxfile)

    # Check if input tiling is requested
    original_files = args.point_cloud.copy()
    all_output_files = []
    
    if args.tile_size > 0:
        print(f"Input tiling enabled: {args.tile_size}m tiles with {args.tile_overlap*100:.1f}% overlap")        
        tiled_files = []
        for point_cloud_file in args.point_cloud:
            tile_files = tile_point_cloud(point_cloud_file, args.tile_size, args.tile_overlap)
            tiled_files.extend(tile_files)
        
        print(f"Processing {len(tiled_files)} tiles...")
        args.point_cloud = tiled_files

    for point_cloud_file in args.point_cloud:

        '''
        Handle input and output file paths-----------------------------------------------------------------------------------
        '''
        
        path = OP.dirname(point_cloud_file)
        file = OP.splitext(OP.basename(point_cloud_file))[0] + "_ours.ply"
        args.odir = OP.join(path, file)

        '''
        Preprocess data into voxels------------------------------------------------------------------------------------------
        '''

        if args.verbose: print('\n----- Preprocessing started -----')

        os.makedirs(args.vxfile, exist_ok=True)
        args.pc, args.headers = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
        args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc)
        
        print(f'Voxelising to {args.grid_size} grid sizes')
        preprocess(args)
        
        if args.verbose:
            print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
            print(f'runtime: {(datetime.datetime.now() - start).seconds}')
        
        '''
        Run semantic training------------------------------------------------------------------------------------------------
        '''
        if args.verbose: print('\n----- Semantic segmenation started -----')
        
        SemanticSegmentation(args)
        torch.cuda.empty_cache()

        if os.path.exists(args.vxfile):
            shutil.rmtree(args.vxfile)

        if args.verbose:
            print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
            print(f'runtime: {(datetime.datetime.now() - start).seconds}')
        
        # Collect output files for potential merging
        if args.tile_size > 0:
            base_path = args.odir.replace('_ours.ply', '')
            class_output = base_path + '_classified.ply'
            conf_output = base_path + '_confidence.ply'
            all_output_files.append({
                'classified': class_output,
                'confidence': conf_output,
                'original': point_cloud_file
            })

    # Handle merging if tiling was used
    if args.tile_size > 0 and args.auto_merge:
        print(f"\nMerging {len(all_output_files)} tile results...")
        
        for original_file in original_files:
            # Find all tiles that belong to this original file
            base_name = OP.splitext(OP.basename(original_file))[0]
            classified_tiles = [f['classified'] for f in all_output_files if base_name in f['original']]
            confidence_tiles = [f['confidence'] for f in all_output_files if base_name in f['original']]
            
            if classified_tiles:
                # Create merged output filenames
                output_dir = OP.dirname(original_file)
                merged_class = OP.join(output_dir, f"{base_name}_merged_classified.ply")
                merged_conf = OP.join(output_dir, f"{base_name}_merged_confidence.ply")
                
                # Merge tiles
                merge_predictions(classified_tiles, merged_class, remove_tiles=True)
                merge_predictions(confidence_tiles, merged_conf, remove_tiles=True)
                
                print(f"Merged results saved to:")
                print(f"  Classification: {merged_class}")
                print(f"  Confidence: {merged_conf}")
    
    elif args.tile_size > 0:
        print(f"\nTiling completed. {len(all_output_files)} tile results created.")
        print("Use --auto-merge flag to automatically merge results, or merge manually using:")
        print("  python utils/tile_pointcloud.py --merge tile_*_classified.ply --merge-output merged_classified.ply")

if __name__ == '__main__':
    main()
