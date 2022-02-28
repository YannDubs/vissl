#!/usr/bin/env bash

python tools/linear_eval.py --feature-path simclr_dir/features --out-path simclr_dir/eval
python tools/linear_eval.py --feature-path swav_dir/features --out-path swav_dir/eval
python tools/linear_eval.py --feature-path cntr_dir/features --out-path cntr_dir/eval
python tools/linear_eval.py --feature-path slfdstl_dir/features --out-path slfdstl_dir/eval