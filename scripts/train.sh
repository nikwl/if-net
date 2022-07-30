# !/bin/bash

SPLITS=/media/DATACENTER2/nikolas/dev/data/mendnet/splits/v3.val.split_"$1".json

# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PYTHONPATH:`pwd`/data_processing
python train.py \
    -splits $SPLITS \
    -std_dev 0.2 0.015 \
    -res 128 \
    -m ShapeNetPoints \
    -batch_size 10 \
    -num_workers 10 \
    -pointcloud \
    -name "$1"
