#!/usr/bin/env bash

python tools/linear_eval.py --feature-path simclr_dir/features --out-path simclr_dir/eval_bn --is-batchnorm --batch-size 2048 --lr 2.4
python tools/linear_eval.py --feature-path swav_dir/features --out-path swav_dir/eval_bn --is-batchnorm --batch-size 2048 --lr 2.4
python tools/linear_eval.py --feature-path cntr_dir/features --out-path cntr_dir/eval_bn --is-batchnorm --batch-size 2048 --lr 2.4
python tools/linear_eval.py --feature-path dstl_dir/features --out-path dstl_dir/eval_bn --is-batchnorm  --batch-size 2048 --lr 2.4