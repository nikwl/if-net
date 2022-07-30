# !/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`/data_processing

declare -a arr=(
    # "jars" 
    # "bottles" 
    # "mugs" 
    # "airplanes" 
    "chairs" 
    "cars" 
    "tables"
    "sofas"
)


for v in "${arr[@]}"
do
    SPLITS=/media/DATACENTER2/nikolas/dev/data/mendnet/splits/v3.val.split_"$v".json
    python data_processing/boundary_sampling.py \
        -sigma 0.015 \
        -splits $SPLITS
    python data_processing/boundary_sampling.py \
        -sigma 0.2 \
        -splits $SPLITS
    python data_processing/voxelized_pointcloud_sampling.py \
        -res 128 \
        -num_points 3000 \
        -splits $SPLITS
done

