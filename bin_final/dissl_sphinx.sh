#!/usr/bin/env bash

model=$1
port=$2

./dev/launch_slurm.sh \
    "$model"_dir \
    config=pretrain/dstl/"$model" \
    +config/server=sphinxmem \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.DATA.TRAIN.DATA_PATHS=["./data/nlp/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=320 \
    config.DATA.NUM_DATALOADER_WORKERS=28 \
    config.SLURM.PORT_ID=$port \
    config.SLURM.NAME="$model" \
    config.SLURM.MEM_GB=1000 \
    config.SLURM.NUM_CPU_PER_PROC=32 \