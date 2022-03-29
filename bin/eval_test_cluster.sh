#!/usr/bin/env bash

dir="$1"
mkdir -p "$dir"/eval_logs
echo "Evaluating" "$1"

sbatch <<EOT
#!/usr/bin/env bash
#SBATCH --job-name=test_eval_"$dir"
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --qos=nopreemption
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --output="$dir"/eval_logs/slurm-%j.out
#SBATCH --error="$dir"/eval_logs/slurm-%j.err

# prepare your environment here
source ~/.bashrc

# EXTRACT FEATURES
#conda activate vissl
#bin/extract_features.sh "$dir"

# LINEAR EVAL
conda activate probing
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalw3l03 --weight-decay 3e-6 --lr 0.3 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalw5l03 --weight-decay 5e-6 --lr 0.3 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalw7l03 --weight-decay 7e-6 --lr 0.3 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalw3l03 --weight-decay 3e-6 --lr 0.3 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalw3l003 --weight-decay 3e-6 --lr 0.03 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalw3l1 --weight-decay 3e-6 --lr 1 --is-no-progress-bar
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalw3l01 --weight-decay 3e-6 --lr 0.1 --is-no-progress-bar
EOT