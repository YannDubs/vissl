#!/usr/bin/env bash

model=dissl_e100_d768_m6_r030_convnext_noopt

python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=pretrain/dstl/"$model" \
    config.CHECKPOINT.DIR="./test/checkpoints" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.SLURM.USE_SLURM=False \
    config.DATA.TRAIN.DATA_LIMIT=10000 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=24 \
    config.DATA.TRAIN.DATA_PATHS=["./data/nlp/imagenet_100/train"] \
    config.DATA.NUM_DATALOADER_WORKERS=0 \
    config.OPTIMIZER.num_epochs=3
