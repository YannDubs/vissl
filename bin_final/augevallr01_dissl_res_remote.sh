#!/usr/bin/env bash

model_name=dissl_res
base_dir="$model_name"_dir

./dev/launch_slurm.sh \
    $base_dir/augeval_l01 \
    config=benchmark/linear_image_classification/imagenet1k/eval_resnet_in1k_linear_largelr \
    +config/server=remote_large \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase99.torch \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=196 \
    config.DATA.NUM_DATALOADER_WORKERS=9 \
    config.SLURM.PORT_ID=40090 \
    config.SLURM.NAME=augeval_l01_"$model_name" \
    config.SLURM.MEM_GB=166 \
    config.SLURM.NUM_CPU_PER_PROC=10 \
    config.SLURM.TIME_HOURS=120 \