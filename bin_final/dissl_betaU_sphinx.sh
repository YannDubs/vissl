#!/usr/bin/env bash

./dev/launch_slurm.sh \
    dissl_betaU_dir \
    config=pretrain/dstl/dissl \
    +config/server=sphinx1 \
    config.LOSS.dstl_issl_loss.beta_pM_unif=2.1 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=384 \
    config.DATA.NUM_DATALOADER_WORKERS=48 \
    config.SLURM.PORT_ID=40051 \
    config.SLURM.NAME=dissl_betaU \
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