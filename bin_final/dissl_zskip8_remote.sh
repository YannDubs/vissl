#!/usr/bin/env bash

# WHEN EVALUATING WILL HAVE TO CHANGE THE CONFIG FILE
./dev/launch_slurm.sh \
    dissl_zskip8_dir \
    config=pretrain/dstl/dissl_zskip8 \
    +config/server=sphinx1 \
    config.DISTRIBUTED.NUM_NODES=2 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=196 \
    config.DATA.NUM_DATALOADER_WORKERS=9 \
    config.SLURM.PORT_ID=40045 \
    config.SLURM.NAME=dissl_zskip8 \
    config.SLURM.MEM_GB=166 \
    config.SLURM.NUM_CPU_PER_PROC=10 \
    config.SLURM.TIME_HOURS=120