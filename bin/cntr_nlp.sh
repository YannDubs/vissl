#!/usr/bin/env bash

./dev/launch_slurm.sh \
    config=pretrain/cntr/cntr_resnet \
    +config/server=nlp \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \

