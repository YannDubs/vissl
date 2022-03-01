#!/usr/bin/env bash

python tools/linear_eval.py --feature-path simclr_dir/features --out-path simclr_dir/eval_tune --torch-validate-param lr batch_size weight_decay is_batchnorm
python tools/linear_eval.py --feature-path swav_dir/features --out-path swav_dir/eval_tune --torch-validate-param lr batch_size weight_decay is_batchnorm
python tools/linear_eval.py --feature-path cntr_dir/features --out-path cntr_dir/eval_tune --torch-validate-param lr batch_size weight_decay is_batchnorm
python tools/linear_eval.py --feature-path slfdstl_dir/features --out-path slfdstl_dir/eval_tune --torch-validate-param lr batch_size weight_decay is_batchnorm