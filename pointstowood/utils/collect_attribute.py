import numpy as np
from pykdtree.kdtree import KDTree
import pandas as pd
from src.io import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default='', type=str, help='point cloud')
    parser.add_argument('--attribute', '-a', default='', type=str, help='point cloud containing attribute of interest')
    parser.add_argument('--attribute_index', default=3, type=int, help='column index of attribute')
    parser.add_argument('--distance', type=float, default=0.05, help='threshold distance between points determining whether attribute taken or not')
    parser.add_argument('--odir', type=str, default='.', help='output directory')

    args = parser.parse_args()

    # Load your two point clouds with XYZ coordinates and labels
    xyz, h = load_file(filename=args.point_cloud, additional_headers=True, verbose=False)
    xyza, ah = load_file(filename=args.attribute, additional_headers=True, verbose=False)

    # Define the KNN algorithm with KDTree
    kd_tree = KDTree(xyza.values[:,:3].astype('float32'))
    distances, indices = kd_tree.query(xyz.values[:,:3].astype('float32'), k=1)

    # Get the attributes for the nearest points
    attributes = xyza.values[indices, args.attribute_index:]

    # Create a mask for distances that are less than or equal to the threshold
    mask = distances <= args.distance

    # Apply the mask to the attributes (this will set the attributes of points with distances greater than the threshold to 0)
    #attributes = attributes * mask[:, None]
    attributes = np.where(mask[:, None], attributes, -999)

    # Convert attributes to a DataFrame
    attribute_df = pd.DataFrame(attributes, columns=ah)

    # Attach the attributes to point_cloud
    xyz = pd.concat([xyz, attribute_df], axis=1)
    xyz = xyz[~(xyz[ah] == -999).any(axis=1)]

    odir = os.path.splitext(args.point_cloud)[0] + '_attribute.ply'

    # Save the resulting point cloud with labels to a file
    save_file(odir,  xyz, additional_fields=h+ah)


