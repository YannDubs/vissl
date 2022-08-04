#!/usr/bin/env bash

EXP_ROOT_DIR="$1"
SFFX="$2"
DATA="$3"
CKPT_DIR="$EXP_ROOT_DIR/checkpoints/"
PARAMS_FILE=$(python -c "from vissl.utils.checkpoint import get_checkpoint_resume_files; print(get_checkpoint_resume_files('"$CKPT_DIR"'))")
BASE_PARAMS=$(basename "$PARAMS_FILE" .torch)
OUT_DIR="/scr/biggest/yanndubs/$EXP_ROOT_DIR/features/$DATA/$BASE_PARAMS"

mkdir -p $OUT_DIR

if  [[ "$DATA" == "sun397" ]]; then
  python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=feature_extraction/extract_resnet_nocrop"$SFFX" \
    config.DATA.TRAIN.DATASET_NAMES=["$DATA"_filelist] \
    config.DATA.TRAIN.DATA_PATHS=["./data/biggest/$DATA/train_images.npy"] \
    config.DATA.TRAIN.LABEL_PATHS=["./data/biggest/$DATA/train_labels.npy"] \
    config.DATA.TEST.DATASET_NAMES=["$DATA"_filelist] \
    config.DATA.TEST.DATA_PATHS=["./data/biggest/$DATA/test_images.npy"] \
    config.DATA.TEST.LABEL_PATHS=["./data/biggest/$DATA/test_labels.npy"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="$CKPT_DIR""$PARAMS_FILE" \
    config.EXTRACT_FEATURES.OUTPUT_DIR="$OUT_DIR" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.SLURM.USE_SLURM=False
elif  [[ "$DATA" == "cifar10" || "$DATA" == "cifar100" ]]; then

  if  [[ "$DATA" == "cifar10" ]]; then
    dataset_dir="cifar-10-batches-py"
    dataset_name="CIFAR10"
  elif [[ "$DATA" == "cifar100" ]]; then
    dataset_dir="cifar-100-batches-py"
    dataset_name="CIFAR100"
  fi

  python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=feature_extraction/extract_resnet_nocrop"$SFFX" \
    config.DATA.TEST.DATASET_NAMES=["$dataset_name"] \
    config.DATA.TRAIN.DATASET_NAMES=["$dataset_name"] \
    config.DATA.TRAIN.DATA_PATHS=["./data/biggest/$dataset_dir/"] \
    config.DATA.TEST.DATA_PATHS=["./data/biggest/$dataset_dir/"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="$CKPT_DIR""$PARAMS_FILE" \
    config.EXTRACT_FEATURES.OUTPUT_DIR="$OUT_DIR" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.SLURM.USE_SLURM=False
else
  python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=feature_extraction/extract_resnet"$SFFX" \
    config.DATA.TEST.DATASET_NAMES=["$DATA"_folder] \
    config.DATA.TRAIN.DATASET_NAMES=["$DATA"_folder] \
    config.DATA.TRAIN.DATA_PATHS=["./data/biggest/$DATA/train"] \
    config.DATA.TEST.DATA_PATHS=["./data/biggest/$DATA/test"] \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="$CKPT_DIR""$PARAMS_FILE" \
    config.EXTRACT_FEATURES.OUTPUT_DIR="$OUT_DIR" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.SLURM.USE_SLURM=False
fi