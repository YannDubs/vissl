#!/usr/bin/env bash

model=dissl_zdim8_nomulti_noema_cp

./dev/launch_slurm.sh \
    "$model"_dir \
    config=pretrain/dstl/"$model" \
    +config/server=sphinx1_std \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=320 \
    config.DATA.NUM_DATALOADER_WORKERS=31 \
    config.SLURM.PORT_ID=40061 \
    config.SLURM.NAME="$model" \
    config.SLURM.MEM_GB=1000 \
    config.SLURM.NUM_CPU_PER_PROC=32 \