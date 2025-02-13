#!/usr/bin/env bash

./dev/launch_slurm.sh \
    swav_queuelong_dir \
    config=pretrain/swav/swav_queuelong \
    +config/server=sphinx1 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=384 \
    config.DATA.NUM_DATALOADER_WORKERS=31 \
    config.SLURM.PORT_ID=40061 \
    config.SLURM.NAME=swav_queuelong \
    config.SLURM.MEM_GB=1000 \
    config.SLURM.NUM_CPU_PER_PROC=32 \

#./dev/launch_slurm.sh \
#    dissl_long_dir \
#    config=pretrain/dstl/dissl_long \
#    +config/server=sphinx1 \
#    config.DISTRIBUTED.NUM_NODES=1 \
#    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
#    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
#    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=384 \
#    config.DATA.NUM_DATALOADER_WORKERS=31 \
#    config.SLURM.PORT_ID=40061 \
#    config.SLURM.NAME=dissl_long \
#    config.SLURM.MEM_GB=1000 \
#    config.SLURM.NUM_CPU_PER_PROC=32 \

