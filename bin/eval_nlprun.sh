#!/usr/bin/env bash

dir="$1"
sffx="$2"
mkdir -p "$dir"/eval_logs
echo "Evaluating" "$dir" "$sffx"

sbatch <<EOT
#!/usr/bin/env zsh
#SBATCH --job-name=eval_"$dir""$sffx"
#SBATCH --partition=jag-hi
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --exclude=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard20,jagupard30,jagupard31
#SBATCH --mem=32G
#SBATCH --output="$dir"/eval_logs/slurm-%j.out
#SBATCH --error="$dir"/eval_logs/slurm-%j.err

#works: jagupard21,
# prepare your environment here
source ~/.zshrc

# EXTRACT FEATURES
conda activate myvissl
bin/extract_features_sphinx.sh "$dir" "$sffx"

# LINEAR EVAL
conda activate probing
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b4096 --weight-decay 1e-6 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w3e-6_l01_b4096 --weight-decay 3e-6 --lr 0.1 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b4096_e300 --weight-decay 1e-6 --lr 0.1 --batch-size 4096 --n-epochs 300 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l01_bn_4096 --weight-decay 1e-5 --lr 0.1 --is-batchnorm --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l01_bn_4096 --weight-decay 1e-4 --lr 0.1 --is-batchnorm --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l01_bn_4096_e300  --weight-decay 1e-4 --lr 0.1 --is-batchnorm --n-epochs 300 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l03_bn_4096_e300  --weight-decay 1e-4 --lr 0.3 --is-batchnorm --n-epochs 300 --batch-size 4096 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l01_b2048 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l01_b2048 --weight-decay 1e-5 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l01_b2048 --weight-decay 1e-4 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-7_l03_b2048 --weight-decay 1e-7 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l03_b2048 --weight-decay 1e-6 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l03_b2048 --weight-decay 1e-5 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l1_b2048 --weight-decay 1e-6 --lr 1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-6_l003_b2048 --weight-decay 1e-6 --lr 0.03 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l03_bn_b2048 --weight-decay 1e-5 --lr 0.3 --is-batchnorm --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l03_bn_b2048 --weight-decay 1e-4 --lr 0.3 --is-batchnorm --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-5_l03_b2048_e300 --weight-decay 1e-5 --lr 0.3 --batch-size 2048 --n-epochs 300 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --feature-path "$dir"/features --out-path "$dir"/eval_w1e-4_l03_b2048_e300 --weight-decay 1e-4 --lr 0.3 --batch-size 2048 --n-epochs 300 --is-no-progress-bar --is-monitor-test
EOT