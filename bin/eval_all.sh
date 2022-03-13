#!/usr/bin/env bash
# conda activate probing
python tools/linear_eval.py --feature-path simclr_dir/features --out-path simclr_dir/eval
python tools/linear_eval.py --feature-path swav_dir/features --out-path swav_dir/eval
python tools/linear_eval.py --feature-path cntr_dir/features --out-path cntr_dir/eval
python tools/linear_eval.py --feature-path dstl_dir/features --out-path dstl_dir/eval
python tools/linear_eval.py --feature-path dstlema_dir/features --out-path dstlema_dir/eval

python tools/linear_eval.py --feature-path simplecntr128_dir/features --out-path simplecntr128_dir/eval
python tools/linear_eval.py --feature-path simplecntr_dir/features --out-path simplecntr_dir/eval
python tools/linear_eval.py --feature-path simplecntr512_dir/features --out-path simplecntr512_dir/eval
python tools/linear_eval.py --feature-path cntr128_dir/features --out-path cntr128_dir/eval


python tools/linear_eval.py --feature-path dstl_dir/features --out-path dstl_dir/eval1 --lr 1
python tools/linear_eval.py --feature-path dstl_dir/features --out-path dstl_dir/evalw5 --weight-decay 5e-6