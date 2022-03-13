#!/usr/bin/env bash

base_dir=swav_dir

./dev/launch_slurm.sh \
    $base_dir/evalvissl \
    config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear
    +config/server=remote \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=3 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase99.torch
#    config.SLURM.PARTITION=interactive \
#    config.SLURM.ADDITIONAL_PARAMETERS.qos=nopreemption \
#    config.SLURM.TIME_HOURS=0 \
#    config.SLURM.TIME_MINUTES=5 \
#    config.SLURM.MEM_GB=16 \
#    config.DATA.TRAIN.DATA_LIMIT=10000 \
#    config.LOG_FREQUENCY=50 \
#    config.OPTIMIZER.num_epochs=7 \