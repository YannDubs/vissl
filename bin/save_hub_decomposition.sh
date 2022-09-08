#!/usr/bin/env bash

dir="$1"
name="$2"
phase="$3"

mkdir -p hub_decomposition  

python extra_scripts/convert_vissl_to_torchvision.py \
    --model_url_or_file issl_pretrained/loss_decomposition/$dir/checkpoints/model_final_checkpoint_phase$phase.torch  \
    --output_dir hub_decomposition \
    --output_name $name.torch

mv hub_decomposition/converted_vissl_$name.torch hub_decomposition/$name.torch