#!/usr/bin/env bash

./dev/launch_slurm.sh \
    dissl_swav_dir \
    config=pretrain/dstl/dissl_swav \
    +config/server=remote_large \
    config.DISTRIBUTED.NUM_NODES=2 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=196 \
    config.DATA.NUM_DATALOADER_WORKERS=9 \
    config.SLURM.PORT_ID=40016 \
    config.SLURM.NAME=dissl_swav \
    config.SLURM.MEM_GB=166 \
    config.SLURM.NUM_CPU_PER_PROC=10 \
    config.SLURM.TIME_HOURS=120