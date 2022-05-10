#!/usr/bin/env bash

model_name=swav_queuelong
base_dir="$model_name"_dir

./dev/launch_slurm.sh \
    $base_dir/augeval \
    config=benchmark/linear_image_classification/imagenet1k/eval_resnet_in1k_linear \
    +config/server=sphinx1 \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet/train"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase399.torch \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=384 \
    config.DATA.NUM_DATALOADER_WORKERS=31 \
    config.SLURM.PORT_ID=40061 \
    config.SLURM.NAME=augeval_"$model_name" \
    config.SLURM.MEM_GB=1000 \
    config.SLURM.NUM_CPU_PER_PROC=32 \