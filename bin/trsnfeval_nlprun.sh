#!/usr/bin/env bash

#SBATCH --exclude=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard17,jagupard20,jagupard30,jagupard31
#SBATCH --nodelist=jagupard29
dir="$1"
sffx="$2"
data="$3"
mkdir -p "$dir"/eval_logs
mkdir -p "$dir"/trnsf/"$data"
echo "Evaluating" "$dir" "$sffx" "on" "$data"
feature_dir=/scr/biggest/yanndubs/"$dir"/features/"$data"

sbatch <<EOT
#!/usr/bin/env zsh
#SBATCH --job-name=eval_"$dir""$sffx"_"$data"
#SBATCH --partition=jag-hi
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=jagupard25
#SBATCH --output="$dir"/eval_logs/slurm-%j.out
#SBATCH --error="$dir"/eval_logs/slurm-%j.err


# prepare your environment here
source ~/.zshrc_nojuice
echo \$(which -p conda)

# EXTRACT FEATURES
echo "Feature directory : $feature_dir"
is_already_features=\$( python -c "from pathlib import Path; print(len(list(Path('"$feature_dir"').glob('**/*chunk0*'))) > 0)" )
echo "is_already_features: \$is_already_features"

if [[ "\$is_already_features" == "True" ]]
then
    echo "Features already present."
else
    echo "featurizing."
    conda activate vissl
    bin/extract_trnsf_features_sphinx.sh "$dir" "$sffx" "$data"
fi

# LINEAR EVAL
echo "Linear eval."
conda activate probing
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-6_l01_b2048 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w3e-6_l01_b2048 --weight-decay 3e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-5_l01_b2048 --weight-decay 1e-5 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w3e-5_l01_b2048 --weight-decay 3e-5 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-4_l01_b2048 --weight-decay 1e-4 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-7_l01_b2048 --weight-decay 1e-7 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-6_l03_b2048 --weight-decay 1e-6 --lr 0.3 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-6_l1_b2048 --weight-decay 1e-6 --lr 1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-6_l003_b2048 --weight-decay 1e-6 --lr 0.03 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-6_l01_b2048_lars --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-lars --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-5_l01_b2048_lars --weight-decay 1e-5 --lr 0.1 --batch-size 2048 --is-lars --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w0_l001_b2048_lars --weight-decay 0 --lr 0.01 --batch-size 2048 --is-lars --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-6_l01_b2048 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-6_l01_b2048_e300 --weight-decay 1e-6 --lr 0.1 --batch-size 2048 --n-epochs 300 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-6_l01_bn_2048 --weight-decay 1e-6 --lr 0.1 --is-batchnorm --batch-size 2048 --is-no-progress-bar --is-monitor-test
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_w1e-5_l01_bn_2048 --weight-decay 1e-5 --lr 0.1 --is-batchnorm --batch-size 2048 --is-no-progress-bar --is-monitor-test

if [[ -f "$dir"/eval ]]; then
    rm -rf "$feature_dir"
fi

EOT