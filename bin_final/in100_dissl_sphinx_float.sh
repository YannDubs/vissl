#!/usr/bin/env bash

model=$1
port=$2

./dev/launch_slurm.sh \
    in100_float_"$model"_dir \
    config=pretrain/dstl/"$model" \
    +config/server=sphinx6 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.DATA.TRAIN.DATA_PATHS=["./data/nlp/imagenet_100/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=144 \
    config.DATA.NUM_DATALOADER_WORKERS=24 \
    config.SLURM.PORT_ID=$port \
    config.SLURM.NAME=in100_float_"$model" \
    config.SLURM.MEM_GB=1000 \
    config.SLURM.NUM_CPU_PER_PROC=32 \
    config.MODEL.AMP_PARAMS.USE_AMP=False