#!/usr/bin/env bash

EXP_ROOT_DIR="$1"
CKPT_DIR="$EXP_ROOT_DIR/checkpoints/"
PARAMS_FILE=$(python -c "from vissl.utils.checkpoint import get_checkpoint_resume_files; print(get_checkpoint_resume_files('"$CKPT_DIR"'))")
BASE_PARAMS=$(basename "$PARAMS_FILE" .torch)
OUT_DIR="$EXP_ROOT_DIR/features/$BASE_PARAMS"

python3 tools/linear_eval.py "$OUT_DIR"