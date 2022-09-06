"""Can be used to evaluate multiple models that are already pretrained.

Use as `python bin/eval_all.py "in100_dissl_e100*_dir" -s _z8192_mask -d data`
"""

import pandas as pd
from pathlib import Path
import argparse
from collect_results_linear import get_results

MAIN_DIR = Path(__file__).absolute().parents[1]

parser = argparse.ArgumentParser(description='Evaluates all desired models.')
parser.add_argument('pattern', type=str, help='pattern of directories to evaluate')
parser.add_argument('--metrics', nargs='+', type=str, default=["accuracy"],  help='pattern of directories to evaluate')
args = parser.parse_args()

best_results = dict()
for p in MAIN_DIR.glob(args.pattern):
    if p.is_dir():
        results = get_results(directory=p,  metrics=args.metrics, is_print=False, is_return=True)
        if results is not None:
            best_results[p.name] = results.loc[["best"]]

best_results = pd.concat(best_results)
best_results.index = best_results.index.droplevel(1)
print(best_results)
