import sys
import pandas as pd
from pathlib import Path

METRICS = ["accuracy","balanced_accuracy"]
DIR = Path(sys.argv[1])
paths = list(DIR.glob("eval_*/model_*/**/train_size_-1/seed_0/all_metrics.csv"))
paths += list(DIR.glob("eval/model_*/**/train_size_-1/seed_0/all_metrics.csv"))

breakpoint()
results = dict()
for p in paths:
    eval_dir = str(p).split("eval_")[1].split("/")[0]
    results[eval_dir] = pd.read_csv(p, index_col=0)[METRICS].T

breakpoint()
all_results = pd.concat(results).droplevel(1)
print(all_results)
all_results.to_csv(DIR/"all_results.csv")

