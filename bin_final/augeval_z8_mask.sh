#!/usr/bin/env bash

base_dir="$1"
epochs="$2"


./dev/launch_slurm.sh \
    $base_dir/augeval \
    config=benchmark/linear_image_classification/imagenet1k/eval_resnet_in1k_linear_z8_mask \
    +config/server=sphinx \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.DATA.TRAIN.DATA_PATHS=["./data/nlp/imagenet256/train"] \
    config.DATA.TEST.DATA_PATHS=["./data/nlp/imagenet256/val"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase"$epochs".torch \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=768 \
    config.DATA.TEST.BATCHSIZE_PER_REPLICA=1536 \
    config.DATA.NUM_DATALOADER_WORKERS=8 \
    config.SLURM.PORT_ID=40005 \
    config.SLURM.NAME=augeval_"$base_dir" \
    config.SLURM.MEM_GB=128 \
    config.SLURM.NUM_CPU_PER_PROC=16 \