#!/usr/bin/env bash

model_name=swav_queuelong
base_dir="$model_name"_dir

#./dev/launch_slurm.sh \
#    $base_dir/trsnfeval/caltech \
#    config=benchmark/linear_image_classification/caltech101/eval_resnet_transfer_caltech_linear \
#    +config/server=sphinx1_2gpu \
#    config.DISTRIBUTED.NUM_NODES=1 \
#    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
#    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase399.torch \
#    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=2048 \
#    config.DATA.NUM_DATALOADER_WORKERS=31 \
#    config.SLURM.PORT_ID=40040 \
#    config.SLURM.MEM_GB=200 \
#    config.SLURM.NUM_CPU_PER_PROC=32 \
#
#
#./dev/launch_slurm.sh \
#    $base_dir/trsnfeval/dtd \
#    config=benchmark/linear_image_classification/dtd/eval_resnet_transfer_dtd_linear \
#    +config/server=sphinx1_2gpu \
#    config.DISTRIBUTED.NUM_NODES=1 \
#    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
#    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase399.torch \
#    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=2048 \
#    config.DATA.NUM_DATALOADER_WORKERS=15 \
#    config.SLURM.PORT_ID=40041 \
#    config.SLURM.MEM_GB=200 \
#    config.SLURM.NUM_CPU_PER_PROC=32 \

./dev/launch_slurm.sh \
    $base_dir/trsnfeval/food101 \
    config=benchmark/linear_image_classification/food101/eval_resnet_transfer_food_linear \
    +config/server=sphinx1_2gpu \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$base_dir/checkpoints/model_final_checkpoint_phase399.torch \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=2048 \
    config.DATA.NUM_DATALOADER_WORKERS=31 \
    config.SLURM.PORT_ID=40042 \
    config.SLURM.MEM_GB=200 \
    config.SLURM.NUM_CPU_PER_PROC=32 \