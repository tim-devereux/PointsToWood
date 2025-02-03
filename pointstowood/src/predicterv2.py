from src.model import Net
import os
import sys
import pandas as pd
import numpy as np
from pykdtree.kdtree import KDTree
from tqdm.auto import tqdm
import torch
from abc import ABC
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from src.io import save_file
from collections import OrderedDict
from numba import jit, prange, set_num_threads
import glob
from torch.utils.data import Sampler
import h5py

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.setrecursionlimit(10 ** 8) 

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.lengths = []
        for key in dataset.keys:
            pc = torch.load(key)
            self.lengths.append(len(pc))
            
        self.indices = np.argsort(self.lengths)
        
    def __iter__(self):
        n = len(self.indices)
        half_batch = self.batch_size // 2
        
        short_samples = self.indices[:n//2].copy()
        long_samples = self.indices[n//2:].copy()
        
        np.random.shuffle(short_samples)
        np.random.shuffle(long_samples)
        
        for i in range(0, len(short_samples) - half_batch + 1, half_batch):
            if i + half_batch <= len(long_samples):
                batch = list(short_samples[i:i + half_batch])
                batch.extend(list(long_samples[i:i + half_batch]))
                np.random.shuffle(batch)  # Shuffle within batch
                yield batch
                
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
class TestingDataset(Dataset, ABC):
    def __init__(self, voxels, max_pts, device, in_memory=False):
        if not voxels:
            raise ValueError("The 'voxels' parameter cannot be empty.")
        self.voxels = voxels
        self.keys = sorted(glob.glob(os.path.join(voxels, '*.pt')))
        self.device = device
        self.max_pts = max_pts
        self.reflectance_index = 3

    def __len__(self):
        return len(self.keys)  

    def __getitem__(self, index):

        point_cloud = torch.load(self.keys[index])
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)

        local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)
        pos = pos - local_shift
        scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max()

        nan_mask = torch.isnan(pos).any(dim=1) | torch.isnan(reflectance)
        pos = pos[~nan_mask]
        reflectance = reflectance[~nan_mask]

        if nan_mask.any(): print(f"Encountered NaN values in sample at index {index}")
        data = Data(pos=pos, reflectance=reflectance, local_shift=local_shift, sf = scaling_factor)
        return data        

def append_to_h5(f, dataset_name, new_data):
    if dataset_name in f:
        dataset = f[dataset_name]
        current_size = dataset.shape[0]
        new_size = current_size + new_data.shape[0]
        dataset.resize(new_size, axis=0)  # Resize along the first dimension
        dataset[current_size:new_size, ...] = new_data  # Append new data
    else:
        max_shape = (None,) + new_data.shape[1:]  # Unlimited rows, fixed other dimensions
        dataset = f.create_dataset(dataset_name, data=new_data, maxshape=max_shape, chunks=True)

from collections import OrderedDict
def load_model(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    adjusted_state_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module.'):
            key = key[7:]
        adjusted_state_dict[key] = value
    model.load_state_dict(adjusted_state_dict, strict=False)
    return model
    
# class PointCloudClassifier:
#     def __init__(self, is_wood, any_wood):
#         self.is_wood = is_wood
#         self.any_wood = any_wood

#     @staticmethod
#     @jit(nopython=True, parallel=True)
#     def compute_labels(nbr_classification, labels, any_wood):
#         num_neighborhoods = labels.shape[0]
#         num_classes = nbr_classification.shape[1]
#         for i in prange(num_neighborhoods):
#             labels[i, 1] = np.median(nbr_classification[i, :, -1])
#             if any_wood != 1:
#                 over_threshold = nbr_classification[i, :, -2] > any_wood
#                 labels[i, 0] = np.where(np.any(over_threshold), 1, 0)  
#             else:
#                 class_votes = np.zeros(num_classes)
#                 for j in range(num_classes):
#                     class_votes[j] = np.sum((nbr_classification[i, :, -2] == j) * nbr_classification[i, :, -1])
#                 labels[i, 0] = np.argmax(class_votes)  
#         return labels

#     def collect_predictions(self, classification, original):
#         original = original.drop(columns=[c for c in original.columns if c in ['label', 'pwood', 'pleaf']])
#         indices_file = os.path.join('nbrs.npy')
        
#         if os.path.exists(indices_file):
#             indices = np.load(indices_file)
#         else:
#             kd_tree = KDTree(classification[:, :3])
#             _, indices = kd_tree.query(original.values[:, :3], k = 32 if self.any_wood != 1 else 64)

#         labels = np.zeros((original.shape[0], 2))
#         labels = self.compute_labels(classification[indices], labels, self.any_wood)
#         original.loc[:, ['label', 'pwood']] = labels
#         return original

from torch_geometric.nn.pool import voxel_grid

class PointCloudClassifier:
    def __init__(self, is_wood, any_wood, voxel_size):
        self.is_wood = is_wood
        self.any_wood = any_wood
        self.voxel_size = voxel_size  

    def classify_voxels(self, data, voxel_indices):
        unique_voxels = torch.unique(voxel_indices)
        labels = torch.zeros_like(voxel_indices, dtype=torch.long)
        pwood = torch.zeros_like(voxel_indices, dtype=torch.float)

        for voxel_id in unique_voxels:
            points_in_voxel = data[voxel_indices == voxel_id]

            if self.any_wood != 1:
                pwood[voxel_indices == voxel_id] = (
                    1 if torch.any(points_in_voxel[:, -2] > self.any_wood) else 0
                )
                labels[voxel_indices == voxel_id] = 0  
            else:
                class_votes = torch.bincount(
                    points_in_voxel[:, -2].long(),  
                    weights=points_in_voxel[:, -1]  
                )
                labels[voxel_indices == voxel_id] = torch.argmax(class_votes)
                pwood[voxel_indices == voxel_id] = torch.max(points_in_voxel[:, -1])

        return labels, pwood

    def collect_predictions(self, classification, original):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classification = torch.tensor(classification).to(device)
        original = torch.tensor(original.values).to(device)

        voxel_indices = voxel_grid(
            pos=classification[:, :3], 
            size=self.voxel_size,
            batch=None
        )

        labels, pwood = self.classify_voxels(classification, voxel_indices)

        original[:, -2] = labels.cpu()
        original[:, -1] = pwood.cpu()

        return original.cpu().numpy()
#########################################################################################################
#                                       SEMANTIC INFERENCE FUNCTION                                     #
#                                       ==========================                                      #

def SemanticSegmentation(args):

    '''
    Setup Multi GPU processing. 
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''
    Setup model. 
    '''

    model = Net(num_classes=1).to(device)

    try:
        load_model(os.path.join(args.wdir,'model',args.model), model, device)
    except KeyError:
        raise Exception(f'No model loaded at {os.path.join(args.wdir,"model",args.model)}')

    #####################################################################################################
    
    '''
    Setup data loader. 
    '''

    test_dataset = TestingDataset(voxels=args.vxfile, device = device, max_pts=args.max_pts)
    
    batch_sampler = BalancedBatchSampler(test_dataset, args.batch_size)
                         
    test_loader = DataLoader(test_dataset, 
                           batch_sampler=batch_sampler,
                           num_workers=0,
                           pin_memory=True)

    #####################################################################################################

    '''
    Initialise model
    '''
    model.eval()

    output_list = []

    if args.transfer_features:
        h5_file = h5py.File('features.h5', 'a')
    else:
        h5_file = None

    with h5py.File('features.h5', 'a') as f:
        with tqdm(total=len(test_loader), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc = "Inference") as pbar:
            for _, data in enumerate(test_loader):

                data = data.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs, features = model(data)
                    outputs = torch.nan_to_num(outputs)

                probs = torch.sigmoid(outputs).to(device)
                preds = (probs>=args.is_wood).type(torch.int64).cpu()
                preds = np.expand_dims(preds, axis=1)

                batches = np.unique(data.batch.cpu())
                pos = data.pos.cpu().numpy()
                probs_2d = np.expand_dims(probs.detach().cpu().numpy(), axis=1)  
                output = np.concatenate((pos, preds, probs_2d), axis=1)
                outputb = None

                for batch in batches:
                    outputb = np.asarray(output[data.batch.cpu() == batch])
                    outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch : 3 + (3 * batch)]
                    output_list.append(outputb)
                
                if args.transfer_features:
                    batch_features = features[data.batch == batch].cpu().detach().numpy()
                    append_to_h5(h5_file, 'pos', outputb[:, :3])
                    append_to_h5(h5_file, 'labels', outputb[:, -2:])
                    append_to_h5(h5_file, 'features', batch_features)
                
                pbar.update(1)
    
        classified_pc = np.vstack(output_list)

        if h5_file is not None:
            h5_file.close()

    #####################################################################################################
    
    '''
    Choosing most confident labels using nearest neighbour search. 
    '''  

    if args.verbose: print("Spatially aggregating prediction probabilites and labels...")
    classifier = PointCloudClassifier(args.is_wood, any_wood=args.any_wood)
    args.pc = classifier.collect_predictions(classified_pc, args.pc)

    '''
    Save final classified point cloud. 
    '''

    headers = list(dict.fromkeys(args.headers+['n_z', 'label', 'pwood']))
    save_file(args.odir, args.pc.copy(), additional_fields= headers, verbose=False)    
    
    return args
