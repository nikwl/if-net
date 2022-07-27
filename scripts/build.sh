# !/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`/data_processing
python data_processing/boundary_sampling.py \
    -sigma 0.015 \
    -splits /media/DATACENTER2/nikolas/dev/data/mendnet/splits/v3.val.split_mugs.json
python data_processing/boundary_sampling.py \
    -sigma 0.2 \
    -splits /media/DATACENTER2/nikolas/dev/data/mendnet/splits/v3.val.split_mugs.json
python data_processing/voxelized_pointcloud_sampling.py \
    -res 128 \
    -num_points 3000 \
    -splits /media/DATACENTER2/nikolas/dev/data/mendnet/splits/v3.val.split_mugs.json