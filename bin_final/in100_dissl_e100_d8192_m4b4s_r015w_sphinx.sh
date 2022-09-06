#!/usr/bin/env bash

model=dissl_e100_d8192_m4b4s_r015w

./dev/launch_slurm.sh \
    in100_"$model"_dir \
    config=pretrain/dstl/"$model" \
    +config/server=sphinx_4 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DATA.TRAIN.DATA_PATHS=["./data/nlp/imagenet_100/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=320 \
    config.DATA.NUM_DATALOADER_WORKERS=15 \
    config.SLURM.PORT_ID=40213 \
    config.SLURM.NAME=in100_"$model" \
    config.SLURM.MEM_GB=480 \
    config.SLURM.NUM_CPU_PER_PROC=18 \

#config.DATA.TRAIN.DATA_PATHS=["/self/scr-sync/nlp/imagenet_100/train"] \