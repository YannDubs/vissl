test_barlow_remote_noslurm.sh

dir="$1"
name="$2"
phase="$3"

python extra_scripts/convert_vissl_to_torchvision.py \
    --model_url_or_file $dir/checkpoints/model_final_checkpoint_phase$phase.torch  \
    --output_dir hub_issl \
    --output_name $name.torch

mv hub_issl/converted_vissl_$name.torch hub_issl/$name.torch