#!/usr/bin/env bash

dir=$1
mkdir "$dir"/eval_logs

#SBATCH --job-name=eval
#SBATCH --partition=p100,t4v1,t4v2,rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output="$dir"/eval_logs/slurm-%j.out
#SBATCH --error="$dir"/eval_logs/slurm-%j.err

# prepare your environment here
source ~/.bashrc

# EXTRACT FEATURES
conda activate vissl
bin/extract_features.sh $dir

# LINEAR EVAL
conda activate probing
python tools/linear_eval.py --feature-path $dir/features --out-path $dir/eval --is-no-progress-bar
python tools/linear_eval.py --feature-path $dir/features --out-path $dir/evalw3l01 --weight-decay 3e-6 --lr 0.1 --is-no-progress-bar
python tools/linear_eval.py --feature-path $dir/features --out-path $dir/evalw5l01 --weight-decay 5e-6 --lr 0.1 --is-no-progress-bar
python tools/linear_eval.py --feature-path $dir/features --out-path $dir/evalw7l01 --weight-decay 7e-6 --lr 0.1 --is-no-progress-bar
python tools/linear_eval.py --feature-path $dir/features --out-path $dir/evalw1l01 --weight-decay 1e-6 --lr 0.1 --is-no-progress-bar
python tools/linear_eval.py --feature-path $dir/features --out-path $dir/evalw3l03 --weight-decay 3e-6 --lr 0.3 --is-no-progress-bar
python tools/linear_eval.py --feature-path $dir/features --out-path $dir/evalw3l005 --weight-decay 3e-6 --lr 0.05 --is-no-progress-bar
python tools/linear_eval.py --feature-path $dir/features --out-path $dir/evalw3l05 --weight-decay 3e-6 --lr 0.5 --is-no-progress-bar