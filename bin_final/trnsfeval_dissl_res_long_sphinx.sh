#!/usr/bin/env bash

model_name=dissl_zdim8_long
base_dir="$model_name"_dir

./dev/launch_slurm.sh \
    $base_dir/trsnfeval \
    config=benchmark/linear_image_classification/caltech101/eval_resnet_transfer_caltech_linear_z8 \
    +config/server=sphinx1_4gpu \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase399.torch \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=196 \
    config.DATA.NUM_DATALOADER_WORKERS=15 \
    config.SLURM.PORT_ID=40092 \
    config.SLURM.NAME=trnsf_"$model_name" \
    config.SLURM.MEM_GB=300 \
    config.SLURM.NUM_CPU_PER_PROC=16 \