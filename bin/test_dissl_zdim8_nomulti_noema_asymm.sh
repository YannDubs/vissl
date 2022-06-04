#!/usr/bin/env bash

python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.CHECKPOINT.DIR="./test_dissl_zdim8_nomulti_noema_asymm/checkpoints" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.SLURM.USE_SLURM=False \
    config.DATA.TRAIN.DATA_LIMIT=100000 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=32 \
    config.OPTIMIZER.num_epochs=3 