#!/usr/bin/env bash

./dev/launch_slurm.sh \
    swav_queue_dir \
    config=pretrain/swav/swav_queue \
    +config/server=remote_large \
    config.DISTRIBUTED.NUM_NODES=2 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=196 \
    config.DATA.NUM_DATALOADER_WORKERS=9 \
    config.SLURM.PORT_ID=40081 \
    config.SLURM.NAME=swav_queue \
    config.SLURM.MEM_GB=82 \
    config.SLURM.NUM_CPU_PER_PROC=10 \
    config.SLURM.TIME_HOURS=1 \