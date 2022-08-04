#!/usr/bin/env bash

#SBATCH --exclude=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard17,jagupard20,jagupard30,jagupard31
#SBATCH --nodelist=jagupard29
dir="$1"
sffx="$2"
data="$3"
mkdir -p "$dir"/eval_logs/"$data"
mkdir -p "$dir"/trnsf/"$data"
echo "Evaluating" "$dir" "$sffx" "on" "$data"
feature_dir=/scr/biggest/yanndubs/"$dir"/features/"$data"

#mkdir -p /u/scr/nlp/data/features_issl/$dir
#cp -r /jagupard25/scr0/yanndubs/dissl_zdim8_long_dir/features/ /u/scr/nlp/data/features_issl/$dir

sbatch <<EOT
#!/usr/bin/env zsh
#SBATCH --job-name=eval_"$dir""$sffx"_"$data"
#SBATCH --partition=john
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodelist=john10
#SBATCH --output="$dir"/eval_logs/"$data"/slurm-%j.out
#SBATCH --error="$dir"/eval_logs/"$data"/slurm-%j.err

mkdir -p /scr/biggest/yanndubs/"$dir"/features/
cp -r  /u/scr/nlp/data/features_issl/"$dir"/features/"$data"/ /scr/biggest/yanndubs/"$dir"/features/

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
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_sklearn_hyper --is-sklearn --is-no-progress-bar --is-validation
python tools/linear_eval.py --no-wandb --feature-path "$feature_dir" --out-path "$dir"/trnsf/"$data"/eval_svm_hyper --is-sklearn --is-no-progress-bar --is-svm --is-validation

if [[ -f "$dir"/eval ]]; then
    rm -rf "$feature_dir"
fi

EOT