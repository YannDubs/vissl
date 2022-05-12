#!/usr/bin/env bash

dir="$1"
sffx="$2"
mkdir -p "$dir"/eval_logs
echo "Evaluating" "$dir" "$sffx"

sbatch <<EOT
#!/usr/bin/env bash
#SBATCH --job-name=eval_"$dir""$sffx"
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
bin/extract_features_remote.sh "$dir" "$sffx"

# LINEAR EVAL
conda activate probing
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b4096 --weight-decay 1e-6 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b4096_lars --weight-decay 1e-6 --lr 0.1 --batch-size 4096 --is-lars --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w0_l001_b4096_lars --weight-decay 0 --lr 0.01 --batch-size 4096 --is-lars --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w3e-6_l01_b4096 --weight-decay 3e-6 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b4096_e300 --weight-decay 1e-6 --lr 0.1 --batch-size 4096 --n-epochs 300 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l01_b4096 --weight-decay 1e-5 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l01_b4096 --weight-decay 1e-4 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w3e-7_l01_b4096 --weight-decay 3e-7 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l03_b4096 --weight-decay 1e-6 --lr 0.3 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l003_b4096 --weight-decay 1e-6 --lr 0.03 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l001_b4096 --weight-decay 1e-6 --lr 0.01 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l1_b4096 --weight-decay 1e-6 --lr 1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval --is-no-progress-bar --is-monitor-test
rm -rf "$dir"/features

EOT