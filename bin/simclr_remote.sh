#!/usr/bin/env bash

./dev/launch_slurm.sh \
    config=pretrain/simclr/simclr_resnet \
    +config/server=remote \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.SLURM.PARTITION=interactive \
    config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
    config.SLURM.TIME_HOURS=2 \
    config.SLURM.MEM_GB=16 \
    #config.LOG_FREQUENCY=50 \
    #config.OPTIMIZER.num_epochs=7 \
    #config.DATA.TRAIN.DATA_LIMIT=3000

