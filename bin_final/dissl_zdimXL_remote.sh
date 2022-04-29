#!/usr/bin/env bash

./dev/launch_slurm.sh \
    dissl_zdimXL_dir \
    config=pretrain/dstl/dissl_zdimXL \
    +config/server=remote \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=96 \
    config.DATA.NUM_DATALOADER_WORKERS=31 \
    config.SLURM.PORT_ID=40060 \
    config.SLURM.NAME=dissl_zdimXL \
    config.SLURM.MEM_GB=128 \
    config.SLURM.NUM_CPU_PER_PROC=4 \
    config.DATA.NUM_DATALOADER_WORKERS=4 \
#    config.SLURM.PARTITION=interactive \
#    config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
#    config.SLURM.TIME_HOURS=0 \
#    config.SLURM.TIME_MINUTES=5 \
#    config.SLURM.MEM_GB=16 \
#    config.DATA.TRAIN.DATA_LIMIT=10000
#    config.LOG_FREQUENCY=50 \
#    config.OPTIMIZER.num_epochs=7 \


# could increase NUM_DATALOADER_WORKERS=60 if use smaller imagenet