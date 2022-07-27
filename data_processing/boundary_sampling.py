import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback
import json

ROOT = 'shapenet/data'


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


def boundary_sampling(path):

    path, break_idx = path
    try:

        # off_path = path + '/isosurf_scaled.off'
        off_path = path + "/model_c.obj"
        # out_file = path +'/boundary_{}_samples.npz'.format(args.sigma)
        out_file = path +'/boundary_{}_{}_samples.npz'.format(args.sigma, break_idx)

        if os.path.exists(out_file):
            return

        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', type=float)
    parser.add_argument('-splits', type=str)

    args = parser.parse_args()

    for key in ["id_train_list", "id_val_list"]:

        paths = get_paths(args.splits, key)

        sample_num = 100000

        p = Pool(mp.cpu_count())
        p.map(boundary_sampling, paths)
