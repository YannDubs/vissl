#!/usr/bin/env bash

dir="$1"
mkdir -p "$dir"/eval_logs
echo "Evaluating" "$1"

sbatch <<EOT
#!/usr/bin/env bash
#SBATCH --job-name=eval_"$dir"
#SBATCH --partition=t4v2,rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output="$dir"/eval_logs/slurm-%j.out
#SBATCH --error="$dir"/eval_logs/slurm-%j.err

# prepare your environment here
source ~/.bashrc

# EXTRACT FEATURES
conda activate vissl
bin/extract_features_remote.sh "$dir"

# LINEAR EVAL
conda activate probing
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b2048 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l01_b2048 --weight-decay 1e-5 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l01_b2048 --weight-decay 1e-4 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-7_l03_b2048 --weight-decay 1e-7 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l03_b2048 --weight-decay 1e-6 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l03_b2048 --weight-decay 1e-5 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l1_b2048 --weight-decay 1e-6 --lr 1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l003_b2048 --weight-decay 1e-6 --lr 0.03 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b4096 --weight-decay 1e-6 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b4096_e300 --weight-decay 1e-6 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test --n-epochs 300
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_bn_b4096_e300 --weight-decay 1e-6 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test --n-epochs 300 --is-batchnorm
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l01_bn_b2048 --weight-decay 1e-5 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-batchnorm
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l01_bn_b2048 --weight-decay 1e-4 --lr 0.1 --is-batchnorm --batch-size 2048 --is-no-progress-bar --is-monitor-test
EOT