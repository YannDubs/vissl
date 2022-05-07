#!/usr/bin/env bash

./dev/launch_slurm.sh \
    dissl_res2_dir \
    config=pretrain/dstl/dissl_res \
    +config/server=sphinx1_4gpu \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=384 \
    config.DATA.NUM_DATALOADER_WORKERS=31 \
    config.SLURM.PORT_ID=40012 \
    config.SLURM.NAME=dissl_res2 \
    config.SLURM.MEM_GB=500 \
    config.SLURM.NUM_CPU_PER_PROC=32 \