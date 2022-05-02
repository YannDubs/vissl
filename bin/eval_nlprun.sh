#!/usr/bin/env bash

dir="$1"
mkdir -p "$dir"/eval_logs
echo "Evaluating" "$1"

sbatch <<EOT
#!/usr/bin/env zsh
#SBATCH --job-name=eval_"$dir"
#SBATCH --partition=jag-hi
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --exclude=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard20
#SBATCH --nodelist=jagupard21
#SBATCH --mem=32G
#SBATCH --output="$dir"/eval_logs/slurm-%j.out
#SBATCH --error="$dir"/eval_logs/slurm-%j.err

# prepare your environment here
source ~/.zshrc

# EXTRACT FEATURES
conda activate myvissl
bin/extract_features.sh "$dir"

# LINEAR EVAL
conda activate probing
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w0_l1_b2048 --weight-decay 0 --lr 1 --batch-size 2048 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-7_l1_b2048 --weight-decay 1e-7 --lr 1 --batch-size 2048 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l1_b2048 --weight-decay 1e-6 --lr 1 --batch-size 2048 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l1_b2048 --weight-decay 1e-5 --lr 1 --batch-size 2048 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l1_b2048 --weight-decay 1e-4 --lr 1 --batch-size 2048 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l03_b2048 --weight-decay 1e-6 --lr 0.3 --batch-size 2048 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l3_b2048 --weight-decay 1e-6 --lr 3 --batch-size 2048 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b2048 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l1_bn_b2048 --weight-decay 1e-6 --lr 1 --is-batchnorm --batch-size 2048 --is-no-progress-bar
EOT