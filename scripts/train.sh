# !/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`/data_processing
python train.py \
    -splits /media/DATACENTER2/nikolas/dev/data/mendnet/splits/v3.val.split_mugs.json \
    -std_dev 0.2 0.015 \
    -res 128 \
    -m ShapeNetPoints \
    -batch_size 10 \
    -num_workers 10 \
    -pointcloud
