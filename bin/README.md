# Useful scripts

- `bin/eval_all.py`: evaluate all non-evaluated pretrained models using `bin/eval_nlprun.sh`. Example Use as `python bin/eval_all.py "in100_dissl_e100*_dir" -s _z8192_mask -d data`
- `bin/eval_nlprun.sh`: evaluates a single pretrained model by first featurizing with `bin/extract_features_sphinx.sh` and then linear probing with `tools/linear_eval.py`. Example `bin/eval_nlprun.sh in100_dissl_e100_d8192_m6_r050wq_dir _z8192_mask imagenet_100`
- 