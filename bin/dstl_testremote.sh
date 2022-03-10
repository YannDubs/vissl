#!/usr/bin/env bash

./dev/launch_slurm.sh \
    test_dstl_dir \
    config=pretrain/dstl/dstl_resnet \
    +config/server=remote \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=False \
    config.SLURM.MEM_GB=64 \
    config.SLURM.TIME_HOURS=1 \
    config.SLURM.PARTITION=interactive \
    config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
#    config.DATA.TRAIN.DATA_LIMIT=10000
#    config.LOG_FREQUENCY=50 \
#    config.OPTIMIZER.num_epochs=7 \


