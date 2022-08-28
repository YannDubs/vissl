#!/usr/bin/env bash

base_dir="$1"
epochs="$2"


./dev/launch_slurm.sh \
    $base_dir/augeval_barlow \
    config=benchmark/linear_image_classification/imagenet1k/eval_resnet_in1k_linear_barlow \
    +config/server=sphinx \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.DATA.TEST.DATA_PATHS=["./data/imagenet/val"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase"$epochs".torch \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=1024 \
    config.DATA.TEST.BATCHSIZE_PER_REPLICA=2048 \
    config.DATA.NUM_DATALOADER_WORKERS=15 \
    config.SLURM.PORT_ID=40056 \
    config.SLURM.NAME=augevalbt_"$base_dir" \
    config.SLURM.MEM_GB=128 \
    config.SLURM.NUM_CPU_PER_PROC=16 \