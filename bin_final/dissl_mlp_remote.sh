#!/usr/bin/env bash

./dev/launch_slurm.sh \
    dissl_mlp_dir \
    config=pretrain/dstl/dissl_mlp \
    +config/server=nlp_XL \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=196 \
    config.DATA.NUM_DATALOADER_WORKERS=11 \
    config.SLURM.PORT_ID=40090 \
    config.SLURM.NAME=dissl_mlp \
    config.SLURM.MEM_GB=110 \
    config.SLURM.NUM_CPU_PER_PROC=12 \