#!/usr/bin/env bash

EXP_ROOT_DIR="$1"
CKPT_DIR="$EXP_ROOT_DIR/checkpoints/"
PARAMS_FILE=$(python -c "from vissl.utils.checkpoint import get_checkpoint_resume_files; print(get_checkpoint_resume_files('"$CKPT_DIR"'))")

python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=feature_extraction/extract_features \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="$PARAMS_FILE" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.SLURM.USE_SLURM=False \