#!/usr/bin/env bash

model=$1
port=$2

./dev/launch_slurm.sh \
    cont_"$model"_dir \
    config=pretrain/dstl/"$model" \
    +config/server=sphinx6_hi \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.DATA.TRAIN.DATA_PATHS=["./data/nlp/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256 \
    config.DATA.NUM_DATALOADER_WORKERS=24 \
    config.DATA.IS_NOWARMUP=false \
    config.SLURM.PORT_ID=$port \
    config.SLURM.NAME=cont_"$model" \
    config.SLURM.MEM_GB=1000 \
    config.SLURM.NUM_CPU_PER_PROC=32 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=to_cont.torch \

    # should really be that but for now let's make sure it works first
    #config.MODEL.TRUNK.WEIGHTS_INIT.PARAMS_FILE="$model"_dir/checkpoints/model_final_checkpoint_phase99.torch

