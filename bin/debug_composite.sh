#!/usr/bin/env bash

#source /sailhome/yanndubs/anaconda3/etc/profile.d/conda.sh
#conda activate vissl

python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config=pretrain/dstl/dissl_res \
    config.CHECKPOINT.DIR="./debug_cpmposite2/checkpoints" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.SLURM.USE_SLURM=False \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=384 \
    config.DATA.NUM_DATALOADER_WORKERS=31