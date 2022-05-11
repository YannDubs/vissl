#!/usr/bin/env bash

dir="$1"
sffx="$2"
mkdir -p "$dir"/evalmlp_logs
echo "Evaluating" "$dir" "$sffx"

sbatch <<EOT
#!/usr/bin/env zsh
#SBATCH --job-name=evalmlp_"$dir""$sffx"
#SBATCH --partition=jag-hi
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --exclude=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard20,jagupard30,jagupard31
#SBATCH --mem=32G
#SBATCH --output="$dir"/evalmlp_logs/slurm-%j.out
#SBATCH --error="$dir"/evalmlp_logs/slurm-%j.err

#works: jagupard25,21,26,22,28,29,16
# prepare your environment here
source ~/.zshrc

# EXTRACT FEATURES
conda activate myvissl
bin/extract_features_sphinx.sh "$dir" "$sffx"

# LINEAR EVAL
conda activate probing
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-5_l01_b256 --weight-decay 1e-5 --lr 0.1 --batch-size 256 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-5_l01_b128 --weight-decay 1e-5 --lr 0.1 --batch-size 128 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-5_l01_b64 --weight-decay 1e-5 --lr 0.1 --batch-size 64 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-5_l01_b512 --weight-decay 1e-5 --lr 0.1 --batch-size 512 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-6_l01_b2048 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-5_l01_b2048 --weight-decay 1e-5 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-4_l01_b2048 --weight-decay 1e-4 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-4_l01_b2048 --weight-decay 1e-3 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-5_l03_b2048 --weight-decay 1e-5 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-5_l003_b2048 --weight-decay 1e-5 --lr 0.03 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-4_l01_b2048 --weight-decay 1e-4 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-mlpS
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/evalmlp_w1e-4_l01_b2048 --weight-decay 1e-3 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-mlpS
EOT