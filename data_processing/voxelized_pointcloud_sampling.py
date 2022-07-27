import implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import random
import traceback
import json

ROOT = 'shapenet/data/'


def get_paths(splits, key):
    object_id_dict = json.load(open(splits, "r"))
    paths = []
    for o in object_id_dict[key]:
        for break_idx in range(10):
            path = os.path.join(
                "/media/DATACENTER2/nikolas/dev/data/mendnet/datasets/v3.ShapeNet_r",
                o[0], o[1], "models"
            )
            if os.path.exists(path + "/model_c.obj"):
                paths.append([path, break_idx])
    return paths


def voxelized_pointcloud_sampling(path):

    path, break_idx = path
    try:
        # off_path = path + '/isosurf_scaled.off'
        off_path = path + "/model_c.obj"
        # out_file = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(args.res, args.num_points)
        out_file = path + '/voxelized_point_cloud_{}res_{}_{}points.npz'.format(args.res, args.num_points, break_idx)

        if os.path.exists(out_file):
            print('File exists. Done.')
            return

        mesh = trimesh.load(off_path)
        point_cloud = mesh.sample(args.num_points)


        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)


        np.savez(out_file, point_cloud=point_cloud, compressed_occupancies = compressed_occupancies, bb_min = bb_min, bb_max = bb_max, res = args.res)
        print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)
    parser.add_argument('-splits', type=str)

    args = parser.parse_args()


    bb_min = -0.5
    bb_max = 0.5



    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, args.res)
    kdtree = KDTree(grid_points)

    for key in ["id_train_list", "id_val_list"]:
        paths = get_paths(args.splits, key)
    
        p = Pool(mp.cpu_count())
        # paths = glob(ROOT + '/*/*/')

        # enabeling to run te script multiple times in parallel: shuffling the data
        random.shuffle(paths)
        p.map(voxelized_pointcloud_sampling, paths)