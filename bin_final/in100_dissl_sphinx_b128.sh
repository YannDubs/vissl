#!/usr/bin/env bash

model=$1
port=$2

./dev/launch_slurm.sh \
    in100_b128_"$model"_dir \
    config=pretrain/dstl/"$model" \
    +config/server=sphinx_4 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DATA.TRAIN.DATA_PATHS=["./data/nlp/imagenet_100/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=128 \
    config.DATA.NUM_DATALOADER_WORKERS=30 \
    config.SLURM.PORT_ID=$port \
    config.SLURM.NAME=in100_b128_"$model" \
    config.SLURM.MEM_GB=500 \
    config.SLURM.NUM_CPU_PER_PROC=32 \