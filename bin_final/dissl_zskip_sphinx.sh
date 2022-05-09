#!/usr/bin/env bash

# WHEN EVALUATING WILL HAVE TO CHANGE THE CONFIG FILE
./dev/launch_slurm.sh \
    dissl_zskip_dir \
    config=pretrain/dstl/dissl_zskip \
    +config/server=sphinx1 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=384 \
    config.DATA.NUM_DATALOADER_WORKERS=31 \
    config.SLURM.PORT_ID=40098 \
    config.SLURM.NAME=dissl_zskip \
    config.SLURM.MEM_GB=1000 \
    config.SLURM.NUM_CPU_PER_PROC=32 \