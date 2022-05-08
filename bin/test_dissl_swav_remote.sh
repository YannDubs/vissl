#!/usr/bin/env bash


./dev/launch_slurm.sh \
    test_dissl_swav_dir \
    config=pretrain/dstl/dissl_swav \
    +config/server=remote_large \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=20 \
    config.DATA.NUM_DATALOADER_WORKERS=9 \
    config.SLURM.NAME=test_dissl_swav \
    config.SLURM.NUM_CPU_PER_PROC=10 \
    config.SLURM.PARTITION=interactive \
    config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
    config.SLURM.TIME_HOURS=0 \
    config.SLURM.TIME_MINUTES=30 \
    config.SLURM.MEM_GB=16 \
    config.DATA.TRAIN.DATA_LIMIT=10000 \
    config.LOG_FREQUENCY=50 \
    config.OPTIMIZER.num_epochs=7 \
