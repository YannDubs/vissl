#!/usr/bin/env bash

./dev/launch_slurm.sh \
    simplecntr512_dir \
    config=pretrain/cntr/cntr512_resnet \
    config.LOSS.name=simple_cntr_issl_loss \
    +config/server=remote \
    config.DISTRIBUTED.NUM_NODES=2 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    #config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=False \
    #config.SLURM.MEM_GB=64 \
    #config.SLURM.TIME_HOURS=1 \
    #config.SLURM.PARTITION=interactive \
    #config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
    #config.LOG_FREQUENCY=50 \
    #config.OPTIMIZER.num_epochs=7 \
    #config.DATA.TRAIN.DATA_LIMIT=3000