#!/usr/bin/env bash

./dev/launch_slurm.sh \
    simclr_long_dir \
    config=pretrain/simclr/simclr_resnet \
    +config/server=remote \
    config.DISTRIBUTED.NUM_NODES=2 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.OPTIMIZER.num_epochs=400 \
    config.OPTIMIZER.param_schedulers.lr.lengths=[0.025,0.975] \
#    config.SLURM.PARTITION=interactive \
#    config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
#    config.SLURM.TIME_HOURS=0 \
#    config.SLURM.TIME_MINUTES=5 \
#    config.SLURM.MEM_GB=16 \
#    config.DATA.TRAIN.DATA_LIMIT=10000
#    config.LOG_FREQUENCY=50 \
#    config.OPTIMIZER.num_epochs=7 \


