#!/usr/bin/env bash

./dev/launch_slurm.sh \
    dstl_relreg_zdim_dir \
    config=pretrain/dstl/dstl_relreg_zdim_resnet \
    +config/server=remote \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.SLURM.MEM_GB=128 \
    config.SLURM.NUM_CPU_PER_PROC=4 \
#    config.SLURM.PARTITION=interactive \
#    config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
#    config.SLURM.TIME_HOURS=0 \
#    config.SLURM.TIME_MINUTES=5 \
#    config.SLURM.MEM_GB=16 \
#    config.DATA.TRAIN.DATA_LIMIT=10000
#    config.LOG_FREQUENCY=50 \
#    config.OPTIMIZER.num_epochs=7 \

