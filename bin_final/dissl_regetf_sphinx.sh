#!/usr/bin/env bash

./dev/launch_slurm.sh \
    dissl_regetf_dir \
    config=pretrain/dstl/dissl_reg \
    +config/server=sphinx1 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=384 \
    config.DATA.NUM_DATALOADER_WORKERS=31 \
    config.LOSS.dissl_reg_loss.mode=etf \
    config.SLURM.PORT_ID=40054 \
    config.SLURM.NAME=dissl_regetf \
    config.SLURM.MEM_GB=250 \
    config.SLURM.NUM_CPU_PER_PROC=32 \
#    config.SLURM.PARTITION=interactive \
#    config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
#    config.SLURM.TIME_HOURS=0 \
#    config.SLURM.TIME_MINUTES=5 \
#    config.SLURM.MEM_GB=16 \
#    config.DATA.TRAIN.DATA_LIMIT=10000
#    config.LOG_FREQUENCY=50 \
#    config.OPTIMIZER.num_epochs=7 \


# could increase NUM_DATALOADER_WORKERS=60 if use smaller imagenet