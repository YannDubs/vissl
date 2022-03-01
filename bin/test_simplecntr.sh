#!/usr/bin/env bash


python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config.LOSS.name="simple_cntr_issl_loss" \
    config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATA_PATHS=["./data/imagenet256/train"] \
    config=test/integration_test/quick_cntr \
    config.CHECKPOINT.DIR="./test_simplecntr_dir/checkpoints"