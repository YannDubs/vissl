#!/usr/bin/env bash

#jagupard27,jagupard26,jagupard21
dir="$1"
sffx="$2"
data=${3:-'imagenet256'} # should be imagenet256 or imagenet_100
model=${4:-'resnet'} # should be resnet or convnextS
mkdir -p "$dir"/eval_logs
echo "Evaluating " "$dir" "$sffx" on  "$data"
feature_dir=/scr/biggest/yanndubs/"$dir"/"$data"/features

sbatch <<EOT
#!/usr/bin/env zsh
#SBATCH --job-name=eval_"$dir""$sffx"_"$data"
#SBATCH --partition=jag-hi
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --account=nlp
#SBATCH --mem=48G
#SBATCH --exclude=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18
#SBATCH --output="$dir"/eval_logs/slurm-%j.out
#SBATCH --error="$dir"/eval_logs/slurm-%j.err


# prepare your environment here
source ~/.zshrc_nojuice
echo \$(which -p conda)

# EXTRACT FEATURES
echo "Feature directory : $feature_dir"
end_featurized="$feature_dir/is_featurized"

echo "featurizing."
conda activate vissl
bin/extract_features_sphinx.sh "$dir" "$sffx" "$data" "$model"
touch "\$end_featurized"

# LINEAR EVAL
echo "Linear eval."
conda activate probing
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-5_l01_b2048 --weight-decay 1e-5 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l01_b2048 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w3e-6_l01_b2048 --weight-decay 3e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l03_b2048 --weight-decay 1e-6 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l003_b2048 --weight-decay 1e-6 --lr 0.03 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l001_b2048 --weight-decay 1e-6 --lr 0.01 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l0003_b2048 --weight-decay 1e-6 --lr 0.003 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l01_bn_2048 --weight-decay 1e-6 --lr 0.1 --is-batchnorm --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w0_l001_b2048_lars --weight-decay 0 --lr 0.01 --batch-size 2048 --is-lars --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l001_b2048_lars --weight-decay 1e-6 --lr 0.01 --batch-size 2048 --is-lars --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l01_bn_2048 --weight-decay 1e-6 --lr 0.1 --is-batchnorm --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-2_l1e-3_b2048_adamw --weight-decay 1e-2 --lr 1e-3 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-2_l3e-3_b2048_adamw --weight-decay 1e-2 --lr 3e-3 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-2_l3e-4_b2048_adamw --weight-decay 1e-2 --lr 3e-4 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-2_l1e-4_b2048_adamw --weight-decay 1e-2 --lr 1e-4 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-3_l3e-4_b2048_adamw --weight-decay 1e-3 --lr 3e-4 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-4_l3e-4_b2048_adamw --weight-decay 1e-4 --lr 3e-4 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-1_l3e-4_b2048_adamw --weight-decay 1e-1 --lr 3e-4 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w3e-2_l3e-4_b2048_adamw --weight-decay 3e-2 --lr 3e-4 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w3e-3_l3e-4_b2048_adamw --weight-decay 3e-3 --lr 3e-4 --batch-size 2048 --is-no-progress-bar --is-monitor-test --is-adamw
#python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w3e-5_l01_b2048 --weight-decay 3e-5 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
#python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l1_b2048 --weight-decay 1e-6 --lr 1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
#python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l01_b2048_lars --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-lars --is-no-progress-bar --is-monitor-test
#python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-5_l01_b2048_lars --weight-decay 1e-5 --lr 0.1 --batch-size 2048 --is-lars --is-no-progress-bar --is-monitor-test
#python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w0_l001_b2048_lars --weight-decay 0 --lr 0.01 --batch-size 2048 --is-lars --is-no-progress-bar --is-monitor-test
#python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-6_l01_b2048_e300 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --n-epochs 300 --is-no-progress-bar --is-monitor-test
#python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval_w1e-5_l01_bn_2048 --weight-decay 1e-5 --lr 0.1 --is-batchnorm --batch-size 2048 --is-no-progress-bar --is-monitor-test
#python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/eval --is-no-progress-bar --is-monitor-test

if [[ -f "$dir"/eval ]]; then
    rm -rf "$feature_dir"
fi

EOT