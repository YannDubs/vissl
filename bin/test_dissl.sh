#!/usr/bin/env bash

#source /sailhome/yanndubs/anaconda3/etc/profile.d/conda.sh
#conda activate vissl

python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config=pretrain/dstl/dstl_asym_resnet \
    config.CHECKPOINT.DIR="./test_dissl/checkpoints/" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
    config.SLURM.USE_SLURM=False \
    config.DATA.TRAIN.DATA_LIMIT=10000 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=96 \
    config.OPTIMIZER.num_epochs=3