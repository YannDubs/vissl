#!/usr/bin/env bash


python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=pretrain/dstl/dstl_reg_resnet \
    config.CHECKPOINT.DIR="./test_dstlreg/checkpoints/" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.SLURM.USE_SLURM=False \
    config.DATA.TRAIN.DATA_LIMIT=10000 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=96 \
    config.OPTIMIZER.num_epochs=3