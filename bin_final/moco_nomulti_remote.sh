#!/usr/bin/env bash

model=moco_nomulti

./dev/launch_slurm.sh \
    "$model"_dir \
    config=pretrain/moco/"$model" \
    +config/server=remote_large \
    config.DISTRIBUTED.NUM_NODES=2 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64 \
    config.DATA.NUM_DATALOADER_WORKERS=9 \
    config.SLURM.PORT_ID=40012 \
    config.SLURM.NAME="$model" \
    config.SLURM.MEM_GB=166 \
    config.SLURM.NUM_CPU_PER_PROC=10 \
    config.SLURM.TIME_HOURS=120