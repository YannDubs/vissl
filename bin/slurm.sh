#!/bin/bash
#SBATCH --job-name=eval_dstl

#SBATCH --partition=t4v2,rtx6000

#SBATCH --gres=gpu:1

#SBATCH --qos=normal

#SBATCH --cpus-per-task=6

#SBATCH --mem=32GB

#SBATCH --output=slurm-%j.out

#SBATCH --error=slurm-%j.err

source ~/.bashrc

# prepare your environment here
conda activate probing

# put your command here
python tools/linear_eval.py --feature-path dstl_dir/features --out-path dstl_dir/eval